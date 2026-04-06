"""AI-Augmented Reserving Copilot (single-file prototype).

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


# ------------------------------
# Mock Data Generation
# ------------------------------
def generate_mock_data(seed: int = 42) -> pd.DataFrame:
    """Generate synthetic cumulative paid loss data.

    The pattern follows realistic claim development:
    - Cumulative losses increase with development age.
    - Growth is faster at early maturities and tapers at later maturities.
    - More recent accident years are less developed (right-truncated triangle).

    Args:
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns:
            Accident_Year, Development_Month, Cumulative_Paid_Loss
    """
    rng = np.random.default_rng(seed)

    accident_years = list(range(2020, 2025))
    development_months = [12, 24, 36, 48, 60]

    # Typical paid development proportions toward ultimate by age.
    # Example: at 12 months, ~45% paid; by 60 months, ~100% paid.
    age_to_pct = {
        12: 0.45,
        24: 0.70,
        36: 0.85,
        48: 0.95,
        60: 1.00,
    }

    # Keep trend simple: older AY slightly lower ultimate than newer AY.
    base_ultimate = {
        2020: 9_600_000,
        2021: 10_100_000,
        2022: 10_700_000,
        2023: 11_100_000,
        2024: 11_500_000,
    }

    # Right-truncated observed ages by AY as of an evaluation date.
    max_dev_by_ay = {
        2020: 60,
        2021: 48,
        2022: 36,
        2023: 24,
        2024: 12,
    }

    rows: List[Dict[str, float]] = []
    for ay in accident_years:
        ultimate = base_ultimate[ay] * rng.normal(1.0, 0.03)  # AY-level noise
        for dev in development_months:
            if dev > max_dev_by_ay[ay]:
                continue

            # Add small maturity-specific noise while maintaining monotonicity later.
            maturity_noise = rng.normal(1.0, 0.02)
            cum_paid = ultimate * age_to_pct[dev] * maturity_noise

            rows.append(
                {
                    "Accident_Year": ay,
                    "Development_Month": dev,
                    "Cumulative_Paid_Loss": max(cum_paid, 0.0),
                }
            )

    df = pd.DataFrame(rows).sort_values(["Accident_Year", "Development_Month"])

    # Enforce cumulative monotonicity within AY.
    df["Cumulative_Paid_Loss"] = (
        df.groupby("Accident_Year")["Cumulative_Paid_Loss"].cummax().round(0)
    )

    return df.reset_index(drop=True)


# ------------------------------
# Actuarial Engine
# ------------------------------
def create_triangle(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot claims data into a standard cumulative paid triangle.

    Rows are Accident Year, columns are Development Month.
    """
    triangle = df.pivot_table(
        index="Accident_Year",
        columns="Development_Month",
        values="Cumulative_Paid_Loss",
        aggfunc="sum",
    ).sort_index(axis=0).sort_index(axis=1)

    # Keep display clean for Streamlit.
    return triangle.astype(float)


def calculate_ldfs(triangle: pd.DataFrame) -> Dict[Tuple[int, int], float]:
    """Calculate historical volume-weighted age-to-age LDFs.

    Formula for each age pair (a -> b):
        LDF_(a,b) = sum(CumPaid_b across AY with both ages) / sum(CumPaid_a across AY with both ages)

    Args:
        triangle: Cumulative loss triangle.

    Returns:
        Dictionary keyed by (age_from, age_to), with float LDF values.
    """
    dev_ages = list(triangle.columns)
    ldfs: Dict[Tuple[int, int], float] = {}

    for i in range(len(dev_ages) - 1):
        age_from = dev_ages[i]
        age_to = dev_ages[i + 1]

        valid = triangle[[age_from, age_to]].dropna()
        if valid.empty:
            ldfs[(age_from, age_to)] = 1.0
            continue

        numerator = valid[age_to].sum()
        denominator = valid[age_from].sum()

        ldf = float(numerator / denominator) if denominator > 0 else 1.0
        ldfs[(age_from, age_to)] = ldf

    return ldfs


def calculate_ultimates(
    triangle: pd.DataFrame, selected_ldfs: Dict[Tuple[int, int], float]
) -> pd.DataFrame:
    """Project ultimate losses by AY using Chain Ladder logic with selected LDFs.

    For each AY:
    1) Find latest observed cumulative paid at latest observed development age.
    2) Multiply through remaining selected age-to-age factors to 60 months.

    Args:
        triangle: Cumulative loss triangle.
        selected_ldfs: User-selected or historical LDFs keyed by age pairs.

    Returns:
        DataFrame with latest observed age/value, CDF to ultimate, projected ultimate,
        and indicated reserve.
    """
    dev_ages = sorted(triangle.columns.tolist())

    records: List[Dict[str, float]] = []
    for ay, row in triangle.iterrows():
        observed = row.dropna()
        if observed.empty:
            continue

        latest_age = int(observed.index.max())
        latest_cumulative = float(observed.loc[latest_age])

        # Cumulative development factor from latest age to ultimate (max age in triangle).
        cdf_to_ultimate = 1.0
        if latest_age in dev_ages:
            latest_idx = dev_ages.index(latest_age)
            for j in range(latest_idx, len(dev_ages) - 1):
                key = (dev_ages[j], dev_ages[j + 1])
                cdf_to_ultimate *= float(selected_ldfs.get(key, 1.0))

        projected_ultimate = latest_cumulative * cdf_to_ultimate
        reserve = projected_ultimate - latest_cumulative

        records.append(
            {
                "Accident_Year": int(ay),
                "Latest_Dev_Month": latest_age,
                "Latest_Cumulative_Paid": round(latest_cumulative, 0),
                "CDF_to_Ultimate": round(cdf_to_ultimate, 4),
                "Projected_Ultimate": round(projected_ultimate, 0),
                "Indicated_Reserve": round(reserve, 0),
            }
        )

    return pd.DataFrame(records).sort_values("Accident_Year")


# ------------------------------
# LangChain / LLM helper
# ------------------------------
def _get_openai_api_key() -> str | None:
    """Get OPENAI_API_KEY from Streamlit secrets or environment variable."""
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        # st.secrets may be unavailable in some local setups.
        pass

    return os.getenv("OPENAI_API_KEY")


def ask_reserving_assistant(
    user_question: str,
    triangle: pd.DataFrame,
    ultimates: pd.DataFrame,
    api_key: str,
    model_name: str = "gpt-4o-mini",
) -> str:
    """Send context-rich reserving question to LLM via LangChain ChatOpenAI."""
    triangle_md = triangle.to_markdown()
    ultimates_md = ultimates.to_markdown(index=False)

    system_prompt = (
        "You are a senior chief actuary advising a junior reserving actuary. "
        "Be technically precise, practical, and transparent about assumptions. "
        "Use the provided reserving tables as the primary source of truth. "
        "If a conclusion is uncertain, say so and explain what additional data would help."
    )

    human_prompt = f"""
The junior actuary asks:
{user_question}

Current cumulative paid loss triangle (markdown table):
{triangle_md}

Current projected ultimates table (markdown table):
{ultimates_md}

Please provide:
1) A direct answer.
2) Key actuarial observations from the triangle and ultimate projections.
3) Any cautions or assumptions behind your interpretation.
""".strip()

    llm = ChatOpenAI(model=model_name, temperature=0.2, api_key=api_key)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt),
    ]

    response = llm.invoke(messages)
    return str(response.content)


# ------------------------------
# Streamlit App
# ------------------------------
def main() -> None:
    """Render the AI-Augmented Reserving Copilot Streamlit application."""
    st.set_page_config(page_title="AI-Augmented Reserving Copilot", layout="wide")
    st.title("AI-Augmented Reserving Copilot")
    st.caption("Prototype: deterministic chain ladder + GenAI actuarial copilot")

    # Build model inputs.
    raw_df = generate_mock_data()
    triangle = create_triangle(raw_df)
    historical_ldfs = calculate_ldfs(triangle)

    st.subheader("Cumulative Paid Loss Triangle")
    st.dataframe(triangle.style.format("{:,.0f}"), use_container_width=True)

    st.sidebar.header("LDF Overrides")
    st.sidebar.write("Adjust age-to-age factors used for projections.")

    selected_ldfs: Dict[Tuple[int, int], float] = {}
    for (age_from, age_to), hist_ldf in historical_ldfs.items():
        selected_ldfs[(age_from, age_to)] = st.sidebar.number_input(
            label=f"LDF {age_from}-{age_to}",
            min_value=0.5000,
            max_value=5.0000,
            value=float(round(hist_ldf, 4)),
            step=0.0001,
            format="%.4f",
        )

    ultimates = calculate_ultimates(triangle, selected_ldfs)

    st.subheader("Projected Ultimates (Chain Ladder)")
    st.dataframe(
        ultimates.style.format(
            {
                "Latest_Cumulative_Paid": "{:,.0f}",
                "CDF_to_Ultimate": "{:.4f}",
                "Projected_Ultimate": "{:,.0f}",
                "Indicated_Reserve": "{:,.0f}",
            }
        ),
        use_container_width=True,
    )

    st.markdown("---")
    st.subheader("GenAI Reserving Assistant")
    st.write("Ask questions about adequacy, trends, diagnostics, or assumption risk.")

    api_key = _get_openai_api_key()
    if not api_key:
        st.warning(
            "OPENAI_API_KEY is not set. Add it to Streamlit secrets or environment "
            "variables to enable the assistant."
        )

    user_question = st.text_input(
        "Your question:",
        placeholder="Example: Which accident year appears most under-reserved and why?",
    )

    if st.button("Ask Assistant", type="primary"):
        if not user_question.strip():
            st.info("Please enter a question first.")
        elif not api_key:
            st.warning("Cannot query assistant without OPENAI_API_KEY.")
        else:
            with st.spinner("Thinking like a chief actuary..."):
                try:
                    answer = ask_reserving_assistant(
                        user_question=user_question,
                        triangle=triangle,
                        ultimates=ultimates,
                        api_key=api_key,
                    )
                    st.markdown("#### Assistant Response")
                    st.write(answer)
                except Exception as exc:
                    st.error(f"LLM request failed: {exc}")

    with st.expander("Show raw mock data"):
        st.dataframe(raw_df, use_container_width=True)


if __name__ == "__main__":
    main()
