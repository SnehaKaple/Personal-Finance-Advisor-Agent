"""
Personal Finance Advisor — Agent Core
Agentic AI Capstone Project 2026


Architecture : LangGraph StateGraph (6 nodes)
LLM          : Groq LLaMA 3.3 70B (llama-3.3-70b-versatile)
Knowledge    : In-memory keyword-scored KB (10 domains, India-specific)
Calculators  : EMI (reducing-balance) + SIP (future value) embedded in graph
Memory       : LangGraph MemorySaver — sliding window of last 6 turns
"""

from __future__ import annotations

import os
import re
from typing import List, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

# ══════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE BASE  —  10 India-specific personal finance domains
# ══════════════════════════════════════════════════════════════════════════════

FINANCE_KB: dict[str, str] = {
    "budgeting": """
Budgeting Fundamentals (India):
- 50-30-20 Rule: 50% Needs (rent, groceries, EMIs, utilities), 30% Wants (dining, OTT, travel), 20% Savings & investments.
- Zero-Based Budgeting: Give every rupee a specific job — income minus all allocations equals zero each month.
- Envelope Method: Divide cash into physical or digital envelopes per category to prevent overspending.
- Pay Yourself First: Auto-transfer savings to a separate account on payday, before any discretionary spending.
- Expense Tracking Apps: Walnut, Money Manager, YNAB, or a simple Google Sheets template.
- Review budget monthly; adjust for irregular expenses (insurance renewals, festivals, travel).
- Emergency fund of 3–6 months of essential expenses is the first and most critical savings goal.
- Avoid lifestyle inflation — when income rises, increase savings rate proportionally.
- Sinking Fund: Accumulate monthly for planned large expenses (laptop, vacation, wedding) to avoid debt.
""",

    "investing": """
Investing Fundamentals (India):
- Start early — compounding is most powerful with time. ₹1,000/month at 12% CAGR for 30 years = approx ₹35 lakhs.
- Asset classes: Equity (stocks, mutual funds), Debt (FDs, bonds, PPF), Gold (SGBs, Gold ETFs), Real Estate.
- Nifty 50 historical CAGR: approximately 12–14% over 20-year periods, beating inflation consistently.
- Diversification reduces unsystematic risk — allocate across equity, debt, and gold based on risk profile.
- SIP (Systematic Investment Plan): Fixed monthly investment in mutual funds; benefits from rupee cost averaging.
- Index Funds: Passively track Nifty 50 / Sensex. Expense ratio 0.1–0.2%. Ideal for long-term retail investors.
- PPF (Public Provident Fund): 15-year lock-in, ~7.1% p.a. (government-declared), tax-free, 80C eligible.
- NPS (National Pension System): Market-linked pension with extra ₹50,000 deduction under Section 80CCD(1B).
- Rule of 72: Years to double money = 72 ÷ annual return rate. At 12%: 72/12 = 6 years.
- Never try to time the market — time IN the market consistently beats timing the market.
- Risk Profile: Aggressive (>70% equity), Moderate (40–70%), Conservative (<40% equity).
""",

    "mutual_funds": """
Mutual Funds & SIP (India):
- A mutual fund pools money from many investors and deploys it into a diversified portfolio managed by a SEBI-registered fund manager.
- Types: Equity (high growth/risk), Debt (stable/low risk), Hybrid (balanced), Index (passive, low cost), ELSS (tax-saving).
- SIP: Invest as low as ₹500/month. Auto-debit on a fixed date. Rupee cost averaging reduces volatility impact.
- ELSS (Equity Linked Savings Scheme): Tax-saving MF under Section 80C. 3-year lock-in (shortest among 80C). Best returns among tax-saving instruments.
- Expense Ratio: Annual fee as % of AUM. Index funds: 0.1–0.2%. Actively managed: 0.5–2%. Lower is better.
- NAV (Net Asset Value): Price per unit of a scheme. Published daily by AMFI.
- Exit Load: Redemption fee if sold before a defined period (typically 1% within 1 year for equity funds).
- Direct Plan vs Regular Plan: Direct plan cuts out the distributor — gives 0.5–1% higher annual returns.
- CAGR vs Absolute Return: Always compare mutual fund performance using CAGR for investments over 1 year.
- Lump Sum vs SIP: Lump sum is better in a low market; SIP is better in a volatile/high market (rupee cost averaging).
- Fund Categories per SEBI: Large Cap, Mid Cap, Small Cap, Flexi Cap, Balanced Advantage, Liquid, Gilt, etc.
""",

    "tax_saving": """
Tax Saving in India (FY 2025–26):
- Section 80C: Deduction up to ₹1.5 lakh. Options: PPF, ELSS, EPF, LIC premium, NSC, 5-year tax-saver FD, home loan principal.
- Section 80D: Health insurance premiums — ₹25,000 for self/spouse/children; ₹50,000 if parents are senior citizens.
- Section 24(b): Home loan interest deduction up to ₹2 lakh/year for self-occupied property.
- HRA Exemption: Minimum of (actual HRA received | 50%/40% of basic salary | rent paid minus 10% of basic).
- Section 80CCD(1B): Additional ₹50,000 deduction for NPS contributions — over and above the ₹1.5 lakh 80C limit.
- New vs Old Tax Regime: New regime has lower tax slabs but removes deductions like 80C, HRA, 24(b). Compare using an ITR calculator before choosing.
- TDS (Tax Deducted at Source): Deducted on salary, FD interest >₹40,000/year (₹50,000 for senior citizens), rent >₹50,000/month.
- ITR Filing: File by July 31 of the assessment year. Use ITR-1 for salary income up to ₹50 lakh with no capital gains.
- Capital Gains Tax: STCG on equity (held <1 year) = 15%; LTCG on equity (held >1 year, gains >₹1 lakh) = 10%.
- Form 26AS: Annual tax statement — verify TDS credits before filing ITR.
""",

    "loans_emi": """
Loans and EMI Planning (India):
- EMI Formula: EMI = P × r × (1+r)^n / ((1+r)^n - 1), where P = principal, r = monthly interest rate (annual rate / 12 / 100), n = tenure in months.
- Home Loan: LTV (Loan-to-Value) ratio typically 75–90%. Current rates: 8.5–9.5% p.a. (floating). Tenure up to 30 years.
- Personal Loan: Unsecured, high interest (11–24% p.a.). Use only for genuine emergencies — not lifestyle spending.
- Credit Score (CIBIL): 750+ is ideal. Lower scores attract higher interest rates or loan rejection. Check free on CIBIL/Experian/CRIF.
- EMI Safety Rule: Total EMIs should not exceed 40% of monthly take-home salary.
- Debt Avalanche: Pay off the highest interest rate debt first — minimises total interest paid.
- Debt Snowball: Pay off smallest balance first — psychological wins build momentum.
- Prepayment: Making partial prepayments on home loans significantly reduces total interest. No prepayment penalty on floating rate loans (RBI directive).
- Balance Transfer: Move a high-interest loan to a lower-interest lender to reduce interest burden. Check processing fees.
- Loan Against FD: Borrow up to 90% of FD value at ~1–2% above FD rate — much cheaper than personal loan.
""",

    "insurance": """
Insurance Planning (India):
- Term Insurance: Pure life cover with no maturity benefit. Cover = 10–15x annual income. Buy online for 30–40% lower premiums.
- Whole Life / Endowment / ULIPs: Combine insurance + investment — returns are generally poor (4–6%). Not recommended.
- Health Insurance: Minimum ₹5–10 lakh individual or family floater. Cover hospitalisation, surgery, daycare procedures.
- Critical Illness Rider: Lump-sum payout on diagnosis of listed diseases (cancer, heart attack, stroke). Essential add-on.
- BTIR Strategy (Buy Term, Invest the Rest): Separate insurance from investment for better returns on both.
- Claim Settlement Ratio (CSR): Choose insurers with CSR >95% (e.g., LIC, HDFC Life, ICICI Prudential). Check IRDAI annual report.
- Premium increases with age — buy term and health insurance early (before 30 is ideal).
- Super Top-Up Plan: Extend existing health cover (e.g., employer's ₹3 lakh) cheaply — pays above a threshold.
- Review insurance needs every 3 years or after major life events (marriage, child birth, new home loan).
- Personal Accident Cover: Covers disability and accidental death — very low premium, high utility.
""",

    "emergency_fund": """
Emergency Fund (India):
- Target: 3–6 months of essential monthly expenses (rent, groceries, utilities, EMIs, insurance premiums, transport).
- Salaried employees: 3–4 months sufficient. Freelancers / business owners: 6–12 months recommended.
- Build emergency fund BEFORE investing in equity (except mandatory EPF contribution).
- Where to park: Liquid mutual funds (~6.5–7% returns, T+1 redemption), high-yield savings account, or sweep-in FD.
- Do NOT invest emergency fund in equity, crypto, or long-lock-in products.
- After using the emergency fund, replenish it immediately before resuming normal investment SIPs.
- Recommended liquid funds: Parag Parikh Liquid Fund, HDFC Liquid Fund, ICICI Prudential Liquid Fund (for reference only; not recommendations).
- Rule of Thumb: ₹30,000/month expenses → Emergency Fund target = ₹90,000–₹1,80,000.
""",

    "fixed_deposits": """
Fixed Deposits & Bonds (India):
- FD: Guaranteed, risk-free returns. DICGC-insured up to ₹5 lakh per bank per depositor.
- Current FD Rates (2025): 6.5–7.5% for general public; 7–8% for senior citizens (0.5% extra typically).
- TDS deducted on FD interest if annual interest from a single bank exceeds ₹40,000 (₹50,000 for seniors).
- FD Laddering: Split corpus into multiple FDs with staggered maturity dates (e.g., 1yr, 2yr, 3yr) for liquidity + higher rates.
- Breaking FD early: Interest penalty of 0.5–1%. Use loan against FD instead to avoid penalty.
- RBI Floating Rate Bonds: Sovereign-guaranteed, currently ~8.05% p.a., interest payable semi-annually.
- Government Securities (G-Secs): Sovereign guarantee, yield ~7–7.5%. Buy via RBI Retail Direct or Zerodha Coin.
- Corporate Bonds: Higher returns but carry credit risk. Check credit rating — AAA is safest, AA is acceptable.
- Post Office Schemes: NSC (7.7%), Post Office TD (6.9–7.5%), Senior Citizens Savings Scheme (8.2%), Kisan Vikas Patra — all government-backed.
""",

    "goal_planning": """
Goal-Based Financial Planning (India):
- SMART Goals: Specific, Measurable, Achievable, Relevant, Time-bound.
- Short-term goals (<3 years): Park money in FDs, liquid funds, or RDs (Recurring Deposits).
- Medium-term goals (3–7 years): Hybrid / balanced advantage / debt mutual funds.
- Long-term goals (>7 years): Equity mutual funds, index funds, direct stocks, PPF.
- Retirement Corpus Formula: Annual expenses at retirement × 25 (based on the 4% safe withdrawal rule).
- Example: ₹60,000/month expenses at retirement → Corpus needed = ₹60,000 × 12 × 25 = ₹1.8 crore.
- Child Education Fund: SIP of ₹5,000/month for 15 years at 12% CAGR = approx ₹25 lakh+ for higher education.
- Down Payment for Home: Use RD or short-duration debt funds to systematically accumulate over 3–5 years.
- SIP Step-Up: Increase SIP by 10–15% every year in line with annual salary hike for accelerated corpus building.
- Goal-Specific Investments: Do not mix retirement corpus with child education fund — maintain separate folios / accounts.
""",

    "stock_market": """
Stock Market Basics (India):
- NSE (Nifty 50) and BSE (Sensex 30) are India's two major exchanges, both regulated by SEBI.
- Demat Account: Required to hold shares electronically. Open with SEBI-registered brokers (Zerodha, Groww, Upstox).
- Fundamental Analysis: Evaluate revenue, net profit, EPS, P/E ratio, debt-to-equity, ROE, and management quality.
- Technical Analysis: Uses price charts, moving averages, RSI, MACD to predict short-term price movements.
- Blue-Chip Stocks: Large, stable, dividend-paying companies — TCS, Infosys, HDFC Bank, Reliance, HUL.
- P/E Ratio: Price ÷ EPS. Lower than sector average may indicate undervaluation. Nifty 50 historical average P/E: 20–22.
- Penny Stocks: Highly speculative, prone to manipulation. Avoid completely if risk-averse.
- Long-term investing in quality businesses outperforms short-term trading for 90%+ of retail investors.
- Nifty 50 Index Fund: Best entry point for stock market beginners — diversified, low cost, historically reliable.
- Never invest borrowed money in equities — markets can remain irrational longer than you can remain solvent (Keynes).
- SEBI Regulations: All investment advisors must be SEBI-registered. Beware of unregistered tip providers on Telegram/WhatsApp.
""",
}

FINANCE_TOPICS = list(FINANCE_KB.keys())

# ══════════════════════════════════════════════════════════════════════════════
# STATE SCHEMA
# ══════════════════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    query: str
    route: str
    context: str
    answer: str
    confidence: float
    sources: List[str]
    retry_count: int
    history: List[dict]
    user_name: Optional[str]
    calc_result: Optional[str]


# ══════════════════════════════════════════════════════════════════════════════
# AGENT CLASS
# ══════════════════════════════════════════════════════════════════════════════

class FinanceAdvisorAgent:
    """
    LangGraph-based agentic personal finance advisor.
    Graph topology:
        router → (ADVICE/CALCULATE) → retriever → calculator → answer → evaluator
                                                                             ↓  (low confidence)
               (SKIP) ──────────────────────────────────────────→ answer  retry → evaluator
    """

    def __init__(self) -> None:
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key or api_key == "your_groq_api_key_here":
            raise EnvironmentError(
                "GROQ_API_KEY is not set. Please add your key to the .env file."
            )
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=api_key,
            temperature=0.3,
        )
        self.history: List[dict] = []
        self.user_name: Optional[str] = None
        self.graph = self._build_graph()

    # ── NODE 1: Router ─────────────────────────────────────────────────────────
    def router_node(self, state: AgentState) -> AgentState:
        """Classify query into ADVICE / CALCULATE / SKIP without touching the KB."""
        query = state["query"].strip()
        query_lower = query.lower()

        # Detect user name
        name_match = re.search(
            r"(?:my name is|i am|i'm|call me)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)",
            query,
        )
        if name_match:
            state["user_name"] = name_match.group(1).strip()

        # Greetings / chitchat → SKIP (no KB lookup needed)
        skip_patterns = [
            r"^(hi|hello|hey|good\s+(morning|evening|afternoon|night)|what'?s\s*up|howdy)",
            r"^(thanks|thank\s+you|bye|goodbye|ok|okay|cool|great|nice|awesome|perfect)",
            r"^(who are you|what (are|can) you do|help me|introduce yourself)",
            r"^(my name is|i am|i'm|call me)\s+\w+$",
        ]
        for pat in skip_patterns:
            if re.search(pat, query_lower):
                state["route"] = "SKIP"
                return state

        # Calculator intent → CALCULATE
        calc_patterns = [
            r"calculat|compute|how much (?:is|will|would|does)",
            r"what (?:is|would be|will be) .{0,30}(?:emi|sip|return|interest|amount|corpus|value)",
            r"\d+\s*(?:lakh|lakhs|l\b|k\b|thousand|crore|%|percent|rs\.?|₹)",
            r"emi (?:for|of|on)|sip (?:of|for|returns?)",
            r"compound interest|simple interest|future value|present value|maturity amount",
            r"how many years|double my money|rule of 72",
        ]
        for pat in calc_patterns:
            if re.search(pat, query_lower):
                state["route"] = "CALCULATE"
                return state

        state["route"] = "ADVICE"
        return state

    # ── NODE 2: Context Retriever ──────────────────────────────────────────────
    def retriever_node(self, state: AgentState) -> AgentState:
        """Score each KB topic against the query and return the top 3."""
        if state["route"] == "SKIP":
            return state

        query_lower = state["query"].lower()

        keyword_map: dict[str, list[str]] = {
            "budgeting":      ["budget", "50-30-20", "zero-based", "envelope", "spend", "expense",
                               "save", "saving", "track", "monthly", "income", "outflow", "sinking"],
            "investing":      ["invest", "ppf", "nps", "equity", "return", "compounding", "sip",
                               "nifty", "index fund", "asset", "portfolio", "wealth", "grow"],
            "mutual_funds":   ["mutual fund", "sip", "elss", "nav", "expense ratio", "direct plan",
                               "regular plan", "fund", "amc", "lump sum", "redemption", "folio"],
            "tax_saving":     ["tax", "80c", "hra", "tds", "deduction", "itr", "capital gain",
                               "new regime", "old regime", "section", "80d", "exemption", "refund"],
            "loans_emi":      ["loan", "emi", "home loan", "personal loan", "credit score", "cibil",
                               "debt", "prepayment", "interest", "borrow", "ltv", "tenure"],
            "insurance":      ["insurance", "term", "health cover", "life cover", "premium",
                               "claim", "policy", "critical illness", "ulip", "endowment"],
            "emergency_fund": ["emergency", "liquid", "safety net", "3 month", "6 month",
                               "contingency", "rainy day", "buffer"],
            "fixed_deposits": ["fd", "fixed deposit", "bond", "nsc", "post office", "rbi",
                               "g-sec", "interest rate", "kisan", "corporate bond", "gilt"],
            "goal_planning":  ["goal", "retirement", "child", "education", "house", "plan",
                               "corpus", "target", "future", "milestone", "wealth creation"],
            "stock_market":   ["stock", "share", "sensex", "bse", "nse", "demat", "p/e",
                               "fundamental", "technical", "blue chip", "equity", "trading"],
        }

        scored: list[tuple[int, str]] = []
        for topic, keywords in keyword_map.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            if score > 0:
                scored.append((score, topic))

        scored.sort(reverse=True)
        top_topics = [t for _, t in scored[:3]] if scored else ["investing", "budgeting", "goal_planning"]

        state["context"] = "\n---\n".join(FINANCE_KB[t] for t in top_topics)
        state["sources"] = [t.replace("_", " ").title() for t in top_topics]
        return state

    # ── NODE 3: Calculator ─────────────────────────────────────────────────────
    def calculator_node(self, state: AgentState) -> AgentState:
        """Run embedded EMI / SIP / compound-interest calculations when route is CALCULATE."""
        if state["route"] != "CALCULATE":
            state["calc_result"] = None
            return state

        query = state["query"]
        result_str = ""

        # ── EMI Calculation ──
        emi_match = re.search(
            r"(?:emi|loan)[^₹\d]*(?:₹|rs\.?|inr)?\s*([\d,]+)\s*(?:lakh|lakhs|l\b)?[^%\d]*(\d+(?:\.\d+)?)\s*%[^0-9]*(\d+)\s*year",
            query,
            re.IGNORECASE,
        )
        if emi_match:
            principal = float(emi_match.group(1).replace(",", ""))
            if re.search(r"lakh|lakhs|\bl\b", emi_match.group(0), re.IGNORECASE):
                principal *= 100_000
            annual_rate = float(emi_match.group(2))
            years = int(emi_match.group(3))
            r = annual_rate / (12 * 100)
            n = years * 12
            emi = (principal * r * (1 + r) ** n / ((1 + r) ** n - 1)) if r else principal / n
            total_payment = emi * n
            total_interest = total_payment - principal
            result_str = (
                f"**EMI Calculation Result**\n"
                f"- Principal Amount : ₹{principal:,.0f}\n"
                f"- Annual Interest  : {annual_rate}%\n"
                f"- Tenure           : {years} years ({n} months)\n"
                f"- **Monthly EMI    : ₹{emi:,.2f}**\n"
                f"- Total Payment    : ₹{total_payment:,.2f}\n"
                f"- Total Interest   : ₹{total_interest:,.2f}\n"
            )

        # ── SIP Future Value ──
        sip_match = re.search(
            r"sip[^₹\d]*(?:₹|rs\.?|inr)?\s*([\d,]+)[^%\d]*(\d+(?:\.\d+)?)\s*%[^0-9]*(\d+)\s*year",
            query,
            re.IGNORECASE,
        )
        if sip_match and not result_str:
            monthly = float(sip_match.group(1).replace(",", ""))
            annual_rate = float(sip_match.group(2))
            years = int(sip_match.group(3))
            r = annual_rate / (12 * 100)
            n = years * 12
            fv = monthly * (((1 + r) ** n - 1) / r) * (1 + r)
            invested = monthly * n
            gains = fv - invested
            result_str = (
                f"**SIP Calculator Result**\n"
                f"- Monthly SIP      : ₹{monthly:,.0f}\n"
                f"- Expected Return  : {annual_rate}% p.a.\n"
                f"- Duration         : {years} years\n"
                f"- Total Invested   : ₹{invested:,.0f}\n"
                f"- **Estimated Gains: ₹{gains:,.0f}**\n"
                f"- **Maturity Value : ₹{fv:,.0f}**\n"
            )

        # ── Rule of 72 ──
        r72_match = re.search(r"rule of 72|double.*?(\d+(?:\.\d+)?)\s*%|(\d+(?:\.\d+)?)\s*%.*double", query, re.IGNORECASE)
        if r72_match and not result_str:
            rate_str = r72_match.group(1) or r72_match.group(2)
            if rate_str:
                rate = float(rate_str)
                years_to_double = 72 / rate
                result_str = (
                    f"**Rule of 72 — Doubling Time**\n"
                    f"- Annual Return Rate : {rate}%\n"
                    f"- **Years to Double  : {years_to_double:.1f} years**\n"
                )

        state["calc_result"] = result_str if result_str else None
        return state

    # ── NODE 4: Answer Generator ───────────────────────────────────────────────
    def answer_node(self, state: AgentState) -> AgentState:
        """Generate the final answer using context + conversation history."""
        name_prefix = f"Hi {state.get('user_name')}, " if state.get("user_name") else ""

        # SKIP path — canned responses for greetings
        if state["route"] == "SKIP":
            q = state["query"].lower()
            if any(g in q for g in ("who are you", "what can you do", "introduce", "help me")):
                state["answer"] = (
                    f"{name_prefix}I'm your Personal Finance Advisor, powered by LangGraph and "
                    "Groq LLaMA 3.3 70B. I can help you with:\n\n"
                    "- Budgeting & saving strategies\n"
                    "- Mutual fund & SIP guidance\n"
                    "- Tax saving (80C, HRA, NPS, etc.)\n"
                    "- EMI & SIP calculations\n"
                    "- Insurance, loans, goal planning, stock market basics, and more!\n\n"
                    "Just ask me your finance question. 💰"
                )
            elif any(g in q for g in ("thanks", "thank you")):
                state["answer"] = "You're welcome! Feel free to ask any more finance questions. 😊"
            elif any(g in q for g in ("bye", "goodbye")):
                state["answer"] = "Goodbye! Wishing you financial success! 💰"
            else:
                state["answer"] = (
                    f"{name_prefix}Hello! I'm your Personal Finance Advisor. "
                    "Ask me anything about budgeting, investing, tax saving, loans, "
                    "insurance, or any other money topic! 💰"
                )
            state["confidence"] = 1.0
            state["sources"] = []
            return state

        # Build conversation history string (last 6 turns)
        history_str = ""
        for turn in state["history"][-6:]:
            role = "User" if turn["role"] == "user" else "Advisor"
            history_str += f"{role}: {turn['content']}\n"

        calc_prefix = (state["calc_result"] + "\n\n") if state.get("calc_result") else ""

        system_prompt = (
            "You are an expert Personal Finance Advisor specialising in Indian personal finance.\n"
            "Answer ONLY based on the provided financial knowledge context.\n"
            "Use ₹ for Indian Rupees. Format with bullet points or numbered lists where helpful.\n"
            "For calculations, explain the result in simple language after showing the numbers.\n"
            "If a specific figure is not in the context, provide the general concept or formula.\n"
            "Do NOT fabricate specific stock tips, exact future returns, or predictions.\n"
            "If the context does not cover the question, say so honestly and suggest consulting "
            "a SEBI-registered financial advisor.\n"
            "Keep response concise but complete — aim for 150–300 words."
        )

        user_prompt = (
            f"Conversation History:\n{history_str}\n"
            f"Financial Knowledge Context:\n{state['context']}\n"
            f"User Question: {state['query']}\n\n"
            "Provide a helpful, accurate personal finance answer based strictly on the context above."
        )

        response = self.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        state["answer"] = calc_prefix + response.content
        return state

    # ── NODE 5: Confidence Evaluator ───────────────────────────────────────────
    def evaluator_node(self, state: AgentState) -> AgentState:
        """Self-evaluate answer quality; produce a 0–1 confidence score."""
        if state["route"] == "SKIP":
            state["confidence"] = 1.0
            return state

        eval_prompt = (
            "Rate the quality of this personal finance advice on a scale of 0.0 to 1.0.\n\n"
            f"Question: {state['query']}\n"
            f"Answer: {state['answer']}\n"
            f"Context Used (first 600 chars): {state['context'][:600]}...\n\n"
            "Scoring criteria:\n"
            "- Is the answer grounded in the provided context? (highest weight)\n"
            "- Is it accurate and practical for Indian personal finance?\n"
            "- Does it avoid hallucinated numbers or fabricated specific returns?\n"
            "- Is it helpful and reasonably complete?\n\n"
            "Respond with ONLY a single decimal number between 0.0 and 1.0. Nothing else."
        )
        eval_resp = self.llm.invoke([HumanMessage(content=eval_prompt)])
        try:
            score_match = re.search(r"(?:0?\.\d+|1\.0|0\.0|0|1)", eval_resp.content.strip())
            score = float(score_match.group()) if score_match else 0.75
            state["confidence"] = round(min(max(score, 0.0), 1.0), 2)
        except Exception:
            state["confidence"] = 0.75
        return state

    # ── NODE 6: Retry Controller ───────────────────────────────────────────────
    def retry_node(self, state: AgentState) -> AgentState:
        """Regenerate a higher-quality answer when confidence < 0.6."""
        state["retry_count"] = state.get("retry_count", 0) + 1

        system_prompt = (
            "You are an expert Personal Finance Advisor for India.\n"
            "The previous answer scored low on confidence. Provide a BETTER, more grounded answer.\n"
            "Strictly use ONLY the facts in the provided context. Be specific and factual.\n"
            "Do not introduce any numbers, statistics, or claims not present in the context."
        )
        user_prompt = (
            f"Context:\n{state['context']}\n\n"
            f"Question: {state['query']}\n"
            f"Previous low-confidence answer: {state['answer']}\n\n"
            "Provide a more accurate, context-grounded answer:"
        )
        response = self.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        state["answer"] = response.content
        return state

    # ── Conditional Edge Functions ─────────────────────────────────────────────
    def route_decision(self, state: AgentState) -> str:
        return state["route"]

    def confidence_check(self, state: AgentState) -> str:
        if state["confidence"] < 0.6 and state.get("retry_count", 0) < 2:
            return "retry"
        return "done"

    # ── Build LangGraph ────────────────────────────────────────────────────────
    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)

        workflow.add_node("router",     self.router_node)
        workflow.add_node("retriever",  self.retriever_node)
        workflow.add_node("calculator", self.calculator_node)
        workflow.add_node("answer",     self.answer_node)
        workflow.add_node("evaluator",  self.evaluator_node)
        workflow.add_node("retry",      self.retry_node)

        workflow.set_entry_point("router")

        workflow.add_conditional_edges(
            "router",
            self.route_decision,
            {
                "ADVICE":    "retriever",
                "CALCULATE": "retriever",
                "SKIP":      "answer",
            },
        )
        workflow.add_edge("retriever",  "calculator")
        workflow.add_edge("calculator", "answer")
        workflow.add_edge("answer",     "evaluator")

        workflow.add_conditional_edges(
            "evaluator",
            self.confidence_check,
            {"retry": "retry", "done": END},
        )
        workflow.add_edge("retry", "evaluator")

        return workflow.compile(checkpointer=MemorySaver())

    # ── Public API ─────────────────────────────────────────────────────────────
    def run(self, query: str) -> dict:
        """Process a user query and return answer + metadata."""
        self.history.append({"role": "user", "content": query})

        initial_state: AgentState = {
            "query":       query,
            "route":       "ADVICE",
            "context":     "",
            "answer":      "",
            "confidence":  0.0,
            "sources":     [],
            "retry_count": 0,
            "history":     self.history.copy(),
            "user_name":   self.user_name,
            "calc_result": None,
        }

        config = {"configurable": {"thread_id": "finance-advisor-session"}}
        result = self.graph.invoke(initial_state, config=config)

        answer = result["answer"]
        self.history.append({"role": "assistant", "content": answer})

        if result.get("user_name"):
            self.user_name = result["user_name"]

        return {
            "answer":     answer,
            "route":      result["route"],
            "confidence": result["confidence"],
            "sources":    result["sources"],
            "user_name":  result.get("user_name"),
        }
