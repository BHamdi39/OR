import streamlit as st
import pandas as pd
import numpy as np
import pulp
import networkx as nx
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.optimize import linprog
import nashpy as nash

# ── Page Configuration ──────────────────────────────────────────────────────────
st.set_page_config(page_title="Enhanced Operational Research Toolkit", layout="wide")

# ── Language & Translation Setup ────────────────────────────────────────────────
LANGUAGES = {
    "English": {
        # General
        "app_title": "Enhanced Operational Research Toolkit",
        "nav_title": "OR Toolkit Navigation",
        "nav_header": "Choose a module",
        "homepage": "Homepage",
        # Modules
        "math_opt": "Mathematical Optimization",
        "assignment": "Assignment Problem",
        "transportation": "Transportation Problem",
        "sequencing": "Sequencing Problem",
        "game_theory": "Game Theory",
        "network_an": "Network Analysis",
        "simulation": "Simulation",
        "proj_man": "Project Management",
        "inv_ctrl": "Inventory Control",
        # UI text
        "welcome": "Welcome to the Enhanced Operational Research (OR) Toolkit",
        "welcome_desc": "This comprehensive application includes advanced OR techniques with step-by-step solutions.",
        "modules_included": "Modules Included:",
        "select_module": "Select a module from the sidebar to get started.",
        "solve_button": "Solve",
        "calculate_button": "Calculate",
        # Linear-programming specific
        "lp_solver_desc": "Solve Linear Programming problems with complete Simplex iterations and Big-M method.",
        "problem_type": "Problem Type",
        "maximize": "Maximize",
        "minimize": "Minimize",
        "obj_coeffs": "Objective Function Coefficients (comma-separated)",
        "num_constraints": "Number of Constraints",
        "constraint_coeffs": "Coefficients for Constraint",
        "constraint_rhs": "RHS for Constraint",
        "step1_formulation": "Step 1 – LP Model Formulation",
        "step2_feasible_region": "Step 2 – Feasible Region Visualization",
        "step3_simplex_steps": "Step 3 – Complete Simplex Method Iterations",
        "step4_results": "Step 4 – Optimal Solution",
        "step5_conclusion": "Step 5 – Conclusion & Interpretation",
        # Assignment
        "assignment_desc": "Solve assignment problems using the Hungarian Algorithm.",
        "cost_matrix_input": "Enter Cost Matrix (comma-separated rows):",
        "hungarian_steps": "Hungarian Algorithm Steps",
        # Transportation
        "transport_desc": "Solve transportation problems with supply and demand constraints.",
        "supply_input": "Supply (comma-separated):",
        "demand_input": "Demand (comma-separated):",
        "cost_matrix": "Cost Matrix (rows for sources, columns for destinations):",
        # Sequencing
        "sequencing_desc": "Solve job sequencing using Johnson's Rule for 2-machine problems.",
        "job_times": "Enter job processing times (Job, Machine1, Machine2):",
        # Game theory
        "game_theory_desc": "Analyse 2-player bimatrix games and find Nash equilibria.",
        "player1_matrix": "Player 1 Payoff Matrix:",
        "player2_matrix": "Player 2 Payoff Matrix:",
    }
}

# ── Language selection helpers ──────────────────────────────────────────────────
if "lang" not in st.session_state:
    st.session_state.lang = "English"


def t(key: str) -> str:
    return LANGUAGES[st.session_state.lang].get(key, key)


# ── Sidebar ─────────────────────────────────────────────────────────────────────
st.sidebar.title(t("nav_title"))
st.sidebar.selectbox(
    "Language",
    options=["English"],
    key="lang_select",
    on_change=lambda: st.session_state.update({"lang": "English"}),
)

selection = st.sidebar.radio(
    t("nav_header"),
    [
        t("homepage"),
        t("math_opt"),
        t("assignment"),
        t("transportation"),
        t("sequencing"),
        t("game_theory"),
        t("network_an"),
        t("simulation"),
        t("proj_man"),
        t("inv_ctrl"),
    ],
)


# ════════════════════════════════════════════════════════════════════════════════
#                               CORE ALGORITHMS
# ════════════════════════════════════════════════════════════════════════════════
class BigMSimplexSolver:
    """Two-phase Big-M simplex (tableau style) with full iteration storage."""

    def __init__(self, c, A, b, sense, maximize=True, M=1_000_000):
        self.M = float(M)
        self.maximize_user = maximize
        self.c = np.asarray(c, dtype=float)
        self.A = np.asarray(A, dtype=float)
        self.b = np.asarray(b, dtype=float)
        self.sense = list(sense)
        self.iterations = []

        # Ensure RHS ≥ 0
        for i in range(len(self.b)):
            if self.b[i] < 0:
                self.A[i, :] *= -1
                self.b[i] *= -1
                self.sense[i] = {"<=": ">=", ">=": "<="}.get(
                    self.sense[i], self.sense[i]
                )

        self.obj_sign = 1.0 if self.maximize_user else -1.0
        self._build_initial_tableau()

    # ── Tableau construction ────────────────────────────────────────────────────
    def _build_initial_tableau(self):
        m, n = self.A.shape
        slack = sum(s == "<=" for s in self.sense)
        surplus = sum(s == ">=" for s in self.sense)
        artificial = sum(s in (">=", "=") for s in self.sense)
        total_vars = n + slack + surplus + artificial

        T = np.zeros((m + 1, total_vars + 1))
        T[:m, :n] = self.A
        col = n
        self.basic = []
        self.art_cols = []

        for i, s in enumerate(self.sense):
            if s == "<=":  # slack
                T[i, col] = 1
                self.basic.append(col)
                col += 1
            elif s == ">=":  # surplus + artificial
                T[i, col] = -1
                col += 1
                T[i, col] = 1
                self.basic.append(col)
                self.art_cols.append(col)
                col += 1
            else:  # equality → artificial
                T[i, col] = 1
                self.basic.append(col)
                self.art_cols.append(col)
                col += 1

        # RHS
        T[:m, -1] = self.b

        # Objective row
        obj = np.zeros(total_vars)
        obj[:n] = self.obj_sign * self.c
        obj[self.art_cols] = -self.M
        T[-1, :-1] = -obj

        # Eliminate artificial vars from objective
        for i, bcol in enumerate(self.basic):
            if bcol in self.art_cols:
                T[-1, :] -= T[-1, bcol] * T[i, :]

        self.tableau = T
        self.n_orig = n
        self._store(0, None, None, None, None)

    # ── Iteration helpers ───────────────────────────────────────────────────────
    def _store(self, it, ent, leave, pivot, ratios):
        self.iterations.append(
            dict(
                iteration=it,
                tableau=self.tableau.copy(),
                basic=self.basic.copy(),
                entering=ent,
                leaving=leave,
                pivot=pivot,
                ratios=ratios.copy() if ratios is not None else None,
            )
        )

    def _optimal(self):
        return np.all(self.tableau[-1, :-1] >= -1e-10)

    def _entering(self):
        col = np.argmin(self.tableau[-1, :-1])
        return col if self.tableau[-1, col] < -1e-10 else None

    def _ratios(self, col):
        rhs = self.tableau[:-1, -1]
        col_vals = self.tableau[:-1, col]
        return [
            rhs[i] / col_vals[i] if col_vals[i] > 1e-10 else np.inf
            for i in range(len(rhs))
        ]

    def _pivot(self, r, c):
        self.tableau[r, :] /= self.tableau[r, c]
        for i in range(self.tableau.shape[0]):
            if i != r:
                self.tableau[i, :] -= self.tableau[i, c] * self.tableau[r, :]
        self.basic[r] = c

    def solve(self, max_it=100):
        it = 0
        while it < max_it and not self._optimal():
            col = self._entering()
            if col is None:
                break
            ratios = self._ratios(col)
            if all(r == np.inf for r in ratios):
                self._store(it + 1, col, None, None, ratios)
                break
            row = int(np.argmin(ratios))
            self._store(it + 1, col, self.basic[row], self.tableau[row, col], ratios)
            self._pivot(row, col)
            it += 1
        self._store(it, None, None, None, None)

        # Extract solution
        sol = np.zeros(self.n_orig)
        for i, bcol in enumerate(self.basic):
            if bcol < self.n_orig:
                sol[bcol] = self.tableau[i, -1]

        z = self.tableau[-1, -1]
        if not self.maximize_user:
            z = -z

        infeasible = any(
            self.tableau[i, -1] > 1e-8 and bcol in self.art_cols
            for i, bcol in enumerate(self.basic)
        )
        status = "Infeasible" if infeasible else "Optimal"
        return dict(
            status=status, solution=sol, objective_value=z, iterations=self.iterations
        )


# ── Feasible-region plot (2-vars) ─── CORRECTED VERSION ─────────────────────────
def plot_feasible_region_2d(constraints, obj, is_max, opt_sol):
    if len(obj) != 2:
        return None, None

    x_max = y_max = 20
    xs = np.linspace(0, x_max, 1000)
    fig = go.Figure()
    colors = ["red", "blue", "green", "orange", "purple", "brown"]
    corners = [(0, 0)]

    # Draw each constraint
    for i, c in enumerate(constraints):
        a, b = c["coeffs"][0], c["coeffs"][1]  # Extract individual coefficients
        rhs = c["rhs"]
        op = c["operator"]
        if abs(b) > 1e-10:
            ys = (rhs - a * xs) / b
            mask = (ys >= 0) & (ys <= y_max)
            line_style = dict(color=colors[i % len(colors)], width=3)
            if op == "≥":
                line_style["dash"] = "dot"
            elif op == "=":
                line_style["dash"] = "dashdot"
            fig.add_trace(
                go.Scatter(
                    x=xs[mask],
                    y=ys[mask],
                    mode="lines",
                    name=f"C{i+1}: {a:.1f}x₁ + {b:.1f}x₂ {op} {rhs}",
                    line=line_style,
                )
            )
        if abs(a) > 1e-10:
            xi = rhs / a
            if xi >= 0:
                corners.append((xi, 0))
        if abs(b) > 1e-10:
            yi = rhs / b
            if yi >= 0:
                corners.append((0, yi))

    # Intersection of constraints - FIXED VERSION
    for i in range(len(constraints)):
        for j in range(i + 1, len(constraints)):
            c1, c2 = constraints[i], constraints[j]
            # CORRECTED: Both rows should use [0] and [1] indices properly
            A_mat = np.array(
                [[c1["coeffs"][0], c1["coeffs"][1]], [c2["coeffs"][0], c2["coeffs"][1]]]
            )  # FIXED!
            b_vec = np.array([c1["rhs"], c2["rhs"]])
            try:
                x_int, y_int = np.linalg.solve(A_mat, b_vec)
                if x_int >= -1e-3 and y_int >= -1e-3:
                    feasible = True
                    for c in constraints:
                        lhs = c["coeffs"][0] * x_int + c["coeffs"][1] * y_int
                        if (
                            (c["operator"] == "≤" and lhs > c["rhs"] + 1e-3)
                            or (c["operator"] == "≥" and lhs < c["rhs"] - 1e-3)
                            or (c["operator"] == "=" and abs(lhs - c["rhs"]) > 1e-3)
                        ):
                            feasible = False
                            break
                    if feasible:
                        corners.append((max(x_int, 0), max(y_int, 0)))
            except np.linalg.LinAlgError:
                pass

    # Unique & feasible corners
    corners = list({(round(x, 6), round(y, 6)) for x, y in corners})
    feasible = []
    for x, y in corners:
        ok = True
        for c in constraints:
            lhs = c["coeffs"][0] * x + c["coeffs"][1] * y
            if (
                (c["operator"] == "≤" and lhs > c["rhs"] + 1e-3)
                or (c["operator"] == "≥" and lhs < c["rhs"] - 1e-3)
                or (c["operator"] == "=" and abs(lhs - c["rhs"]) > 1e-3)
            ):
                ok = False
                break
        if ok:
            feasible.append((x, y))

    # Fill feasible region
    if len(feasible) > 2:
        cx = np.mean([p[0] for p in feasible])
        cy = np.mean([p[1] for p in feasible])
        feasible.sort(key=lambda p: np.arctan2(p[1] - cy, p[0] - cx))  # FIXED
        fx = [p[0] for p in feasible] + [feasible[0][0]]  # FIXED
        fy = [p[1] for p in feasible] + [feasible[0][1]]  # FIXED
        fig.add_trace(
            go.Scatter(
                x=fx,
                y=fy,
                fill="toself",
                fillcolor="rgba(0,255,0,0.25)",
                line=dict(color="green", width=2),
                name="Feasible Region",
                hoverinfo="skip",
            )
        )

    # Corner points
    if feasible:
        fig.add_trace(
            go.Scatter(
                x=[p[0] for p in feasible],
                y=[p[1] for p in feasible],
                mode="markers+text",
                text=[f"({p[0]:.1f}, {p[1]:.1f})" for p in feasible],
                textposition="top center",
                marker=dict(color="black", size=9),
                name="Corner Points",
            )
        )

    # Optimal solution
    if opt_sol and len(opt_sol) == 2:
        fig.add_trace(
            go.Scatter(
                x=[opt_sol[0]],
                y=[opt_sol[1]],
                mode="markers+text",
                marker=dict(size=15, color="red", symbol="star"),
                text=f"Optimal ({opt_sol[0]:.2f}, {opt_sol[1]:.2f})",
                textposition="top center",
                name="Optimal Solution",
            )
        )

    fig.update_layout(
        title="Feasible Region (solid ≤, dot ≥, dash-dot =)",
        xaxis=dict(title="x₁", range=[0, x_max]),
        yaxis=dict(title="x₂", range=[0, y_max]),
        width=800,
        height=600,
        showlegend=True,
    )
    return fig, feasible


# ════════════════════════════════════════════════════════════════════════════════
#                               MODULE FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════════
def show_homepage():
    st.title(t("welcome"))
    st.markdown(t("welcome_desc"))
    st.header(t("modules_included"))
    st.markdown(
        f"""
        - **{t("math_opt")}** – Simplex with Big-M (full iterations)  
        - **{t("assignment")}** – Hungarian Algorithm  
        - **{t("transportation")}** – Transportation (VAM + LP)  
        - **{t("sequencing")}** – Johnson's Rule  
        - **{t("game_theory")}** – Bimatrix games / Nash equilibria  
        - **{t("network_an")}** – Shortest-path analysis  
        - **{t("simulation")}** – M/M/1 queuing metrics  
        - **{t("proj_man")}** – PERT/CPM basics  
        - **{t("inv_ctrl")}** – EOQ model  
        """
    )
    st.info(t("select_module"))


# ── Mathematical Optimization ───────────────────────────────────────────────────
def mathematical_optimization():
    st.title(t("math_opt"))
    st.markdown(t("lp_solver_desc"))

    problem_type = st.radio(t("problem_type"), (t("maximize"), t("minimize")))
    obj_in = st.text_input(t("obj_coeffs"), "3, 5")
    big_m = st.number_input("Big M value", min_value=1000, value=1_000_000, step=1000)

    n_constraints = st.number_input(t("num_constraints"), 1, 10, 2, 1)
    constraints_raw = []
    for i in range(n_constraints):
        st.subheader(f"Constraint #{i+1}")
        c1, c2, c3 = st.columns(3)
        coeffs = c1.text_input(
            "Coefficients", "2, 1" if i == 0 else "1, 3", key=f"coeff{i}"
        )
        op = c2.selectbox("Operator", ["≤", "≥", "="], key=f"op{i}")
        rhs = c3.number_input("RHS", value=0.0, step=1.0, format="%.10g", key=f"rhs{i}")
        constraints_raw.append(dict(coeffs=coeffs, operator=op, rhs=rhs))

    if st.button(t("solve_button")):
        try:
            c_vec = [float(x.strip()) for x in obj_in.split(",")]
            n_vars = len(c_vec)

            constraints = []
            senses = []
            for idx, raw in enumerate(constraints_raw):
                coeffs = [float(x.strip()) for x in raw["coeffs"].split(",")]
                if len(coeffs) != n_vars:
                    st.error(
                        f"Constraint {idx+1}: expected {n_vars} coefficients, got {len(coeffs)}"
                    )
                    st.stop()
                constraints.append(
                    dict(coeffs=coeffs, operator=raw["operator"], rhs=float(raw["rhs"]))
                )
                senses.append({"≤": "<=", "≥": ">=", "=": "="}[raw["operator"]])

            # ── Step 1: Display formulation ──────────────────────────────────────
            st.header(t("step1_formulation"))
            sense_txt = "Maximise" if problem_type == t("maximize") else "Minimise"
            st.latex(
                rf"\text{{{sense_txt}}}\;Z = "
                + " + ".join(f"{coef:.1f}x_{i+1}" for i, coef in enumerate(c_vec))
            )
            st.write("**Subject to:**")
            for c in constraints:
                lhs = " + ".join(
                    f"{coef:.1f}x_{i+1}" for i, coef in enumerate(c["coeffs"])
                )
                st.latex(rf"{lhs} \; {c['operator']} \; {c['rhs']}")
            st.latex(r"x_i \ge 0 \quad \forall i")

            # ── Solve with PuLP for verification ────────────────────────────────
            prob_type = (
                pulp.LpMaximize if problem_type == t("maximize") else pulp.LpMinimize
            )
            prob = pulp.LpProblem("LP", prob_type)
            vars_lp = [pulp.LpVariable(f"x{i+1}", lowBound=0) for i in range(n_vars)]
            prob += pulp.lpSum(c_vec[i] * vars_lp[i] for i in range(n_vars))
            for c in constraints:
                expr = pulp.lpSum(c["coeffs"][i] * vars_lp[i] for i in range(n_vars))
                if c["operator"] == "≤":
                    prob += expr <= c["rhs"]
                elif c["operator"] == "≥":
                    prob += expr >= c["rhs"]
                else:
                    prob += expr == c["rhs"]
            prob.solve(pulp.PULP_CBC_CMD(msg=False))
            opt_lp = None
            if prob.status == pulp.LpStatusOptimal:
                opt_lp = [
                    v.varValue if v.varValue is not None else 0.0 for v in vars_lp
                ]

            # ── Step 2: Feasible region (2-vars only) ───────────────────────────
            if n_vars == 2:
                st.header(t("step2_feasible_region"))
                fig, corners = plot_feasible_region_2d(
                    constraints, c_vec, problem_type == t("maximize"), opt_lp
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    if corners:
                        st.subheader("Corner-point evaluation")
                        rows = []
                        for x, y in corners:
                            z = c_vec[0] * x + c_vec[1] * y
                            rows.append(
                                dict(
                                    Point=f"({x:.2f}, {y:.2f})",
                                    x1=f"{x:.2f}",
                                    x2=f"{y:.2f}",
                                    Z=f"{z:.2f}",
                                )
                            )
                        st.dataframe(pd.DataFrame(rows))

            # ── Step 3: Big-M simplex full iterations ───────────────────────────
            st.header(t("step3_simplex_steps"))
            st.info("Complete Big-M Simplex tableaux")
            A = np.array([c["coeffs"] for c in constraints], dtype=float)
            b = np.array([c["rhs"] for c in constraints], dtype=float)
            solver = BigMSimplexSolver(
                c_vec, A, b, senses, maximize=(problem_type == t("maximize")), M=big_m
            )
            result = solver.solve()

            for it in result["iterations"]:
                title = (
                    "Initial"
                    if it["iteration"] == 0
                    else (
                        "Final"
                        if it["entering"] is None
                        else f"Iteration {it['iteration']}"
                    )
                )
                st.subheader(title + " tableau")
                T = it["tableau"]
                headers = ["Basic"] + [f"x{i+1}" for i in range(n_vars)]
                extra = T.shape[1] - 1 - n_vars
                headers += [f"v{j+1}" for j in range(extra)] + ["RHS"]
                basic_names = [
                    (f"x{b+1}" if b < n_vars else f"v{b - n_vars + 1}")
                    for b in it["basic"]
                ] + ["Z"]
                df_T = pd.DataFrame(T, columns=headers[1:])
                df_T.insert(0, "Basic", basic_names)
                st.dataframe(df_T.round(4), hide_index=True, use_container_width=True)

                if it["ratios"] is not None and it["iteration"] > 0:
                    ratios_text = []
                    for i, r in enumerate(it["ratios"]):
                        ratios_text.append(
                            f"Row {i+1}: {r:.3f}" if r != np.inf else f"Row {i+1}: ∞"
                        )
                    st.caption("**Ratio test:** " + " | ".join(ratios_text))

            # ── Step 4: Results ─────────────────────────────────────────────────
            st.header(t("step4_results"))
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Big-M Results")
                st.metric("Status", result["status"])
                if result["status"] == "Optimal":
                    st.metric("Objective", f"{result['objective_value']:.4f}")
                    st.table(
                        pd.DataFrame(
                            {
                                "Variable": [f"x{i+1}" for i in range(n_vars)],
                                "Value": [f"{v:.4f}" for v in result["solution"]],
                            }
                        )
                    )
            with col2:
                st.subheader("PuLP verification")
                st.metric("Status", pulp.LpStatus[prob.status])
                if prob.status == pulp.LpStatusOptimal:
                    st.metric("Objective", f"{pulp.value(prob.objective):.4f}")
                    st.table(
                        pd.DataFrame(
                            {
                                "Variable": [v.name for v in vars_lp],
                                "Value": [
                                    (
                                        f"{v.varValue:.4f}"
                                        if v.varValue is not None
                                        else "0.0000"
                                    )
                                    for v in vars_lp
                                ],
                            }
                        )
                    )

            # ── Step 5: Interpretation ─────────────────────────────────────────
            st.header(t("step5_conclusion"))
            if result["status"] == "Optimal":
                st.success("✅ Optimal solution found!")
                st.markdown(
                    f"""
                **Summary:**
                - Method: Big-M Simplex ({len(result['iterations'])-1} iterations)
                - Solution: {', '.join([f'x{i+1}={v:.3f}' for i, v in enumerate(result['solution'])])}
                - Optimal value: {result['objective_value']:.3f}
                """
                )
            elif result["status"] == "Infeasible":
                st.error(
                    "❌ Problem is infeasible (artificial variables remain positive)."
                )
            else:
                st.warning("⚠️ Algorithm terminated without optimality.")

        except ValueError as ve:
            st.error(f"Input format error: {ve}")
            st.error(
                "Please check that all numeric inputs are valid and comma-separated."
            )
        except Exception as e:
            st.error(f"Unexpected error: {e}")


# ── Assignment Problem ─────────────────────────────────────────────────────────
def assignment_problem():
    st.title(t("assignment"))
    st.markdown(t("assignment_desc"))
    default = "9, 2, 7, 8\n6, 4, 3, 7\n5, 8, 1, 8\n7, 6, 9, 4"
    mtx_in = st.text_area(t("cost_matrix_input"), default, height=100)
    if st.button(t("solve_button")):
        try:
            rows = [
                [float(x.strip()) for x in r.split(",")]
                for r in mtx_in.strip().splitlines()
            ]
            C = np.array(rows)
            st.write("**Cost matrix:**")
            st.dataframe(pd.DataFrame(C))
            r_idx, c_idx = linear_sum_assignment(C)
            cost = C[r_idx, c_idx].sum()
            res = pd.DataFrame(
                {
                    "Worker": [f"Worker {i+1}" for i in r_idx],
                    "Task": [f"Task {j+1}" for j in c_idx],
                    "Cost": C[r_idx, c_idx],
                }
            )
            st.subheader("Optimal Assignment")
            st.dataframe(res)
            st.success(f"**Total minimum cost: {cost:.2f}**")
        except Exception as e:
            st.error(f"Error: {e}")


# ── Transportation Problem ─────────────────────────────────────────────────────
def transportation_problem():
    st.title(t("transportation"))
    st.markdown(t("transport_desc"))
    supply_in = st.text_input(t("supply_input"), "300, 400, 500")
    demand_in = st.text_input(t("demand_input"), "250, 350, 400, 200")
    cost_in = st.text_area(
        t("cost_matrix"), "19, 30, 50, 10\n70, 30, 40, 60\n40,  8, 70, 20", height=100
    )

    if st.button(t("solve_button")):
        try:
            supply = [int(x.strip()) for x in supply_in.split(",")]
            demand = [int(x.strip()) for x in demand_in.split(",")]
            cost_rows = [
                [int(x.strip()) for x in r.split(",")]
                for r in cost_in.strip().splitlines()
            ]
            C = np.array(cost_rows)

            if len(supply) != C.shape[0] or len(demand) != C.shape[1]:
                st.error("Cost matrix dimensions must match supply and demand counts.")
                st.stop()

            # Build LP for balanced case
            total_sup, total_dem = sum(supply), sum(demand)
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Supply", total_sup)
            c2.metric("Total Demand", total_dem)
            c3.metric("Balance", "Balanced" if total_sup == total_dem else "Unbalanced")

            if total_sup != total_dem:
                st.warning(
                    "Problem is unbalanced. For demo purposes, solving as-is (may be infeasible)."
                )

            c = C.flatten()
            A_eq, b_eq = [], []
            # Supply constraints
            for i in range(len(supply)):
                row = np.zeros_like(c)
                row[i * len(demand) : (i + 1) * len(demand)] = 1
                A_eq.append(row)
                b_eq.append(supply[i])
            # Demand constraints
            for j in range(len(demand)):
                row = np.zeros_like(c)
                row[j :: len(demand)] = 1
                A_eq.append(row)
                b_eq.append(demand[j])

            res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
            if res.success:
                alloc = res.x.reshape(C.shape)
                st.subheader("Optimal Allocation")
                st.dataframe(
                    pd.DataFrame(
                        alloc,
                        index=[f"S{i+1}" for i in range(C.shape[0])],
                        columns=[f"D{j+1}" for j in range(C.shape[1])],
                    ).round(2)
                )
                st.success(f"**Minimum transportation cost: {res.fun:.2f}**")
            else:
                st.error("LP solver failed.")
        except Exception as e:
            st.error(f"Error: {e}")


# ── Sequencing (Johnson) ───────────────────────────────────────────────────────
def sequencing_problem():
    st.title(t("sequencing"))
    st.markdown(t("sequencing_desc"))
    default = "A, 5, 2\nB, 1, 6\nC, 9, 7\nD, 3, 8\nE, 10, 4"
    job_in = st.text_area(t("job_times"), default, height=120)
    if st.button(t("solve_button")):
        try:
            jobs = []
            for line in job_in.strip().splitlines():
                name, m1, m2 = [x.strip() for x in line.split(",")]
                jobs.append(dict(name=name, m1=float(m1), m2=float(m2)))

            st.subheader("Johnson's Rule Solution")
            first, last = [], []
            pending = jobs.copy()

            while pending:
                min_job = min(pending, key=lambda j: min(j["m1"], j["m2"]))
                if min_job["m1"] <= min_job["m2"]:
                    first.append(min_job)
                else:
                    last.insert(0, min_job)
                pending.remove(min_job)

            sequence = first + last
            st.success(
                f"**Optimal Sequence:** {' → '.join(j['name'] for j in sequence)}"
            )

            # Show sequence table
            seq_data = []
            for i, job in enumerate(sequence):
                seq_data.append(
                    {
                        "Position": i + 1,
                        "Job": job["name"],
                        "Machine 1 Time": job["m1"],
                        "Machine 2 Time": job["m2"],
                    }
                )
            st.dataframe(pd.DataFrame(seq_data))

        except Exception as e:
            st.error(f"Error: {e}")


# ── Game Theory (display only) ─────────────────────────────────────────────────
def game_theory():
    st.title(t("game_theory"))
    st.markdown(t("game_theory_desc"))
    p1_default = "3, 0\n5, 1"
    p2_default = "3, 5\n0, 1"
    c1, c2 = st.columns(2)
    A_in = c1.text_area(t("player1_matrix"), p1_default, height=80)
    B_in = c2.text_area(t("player2_matrix"), p2_default, height=80)

    if st.button(t("solve_button")):
        try:

            def to_mtx(s):
                return np.array(
                    [
                        [float(x.strip()) for x in r.split(",")]
                        for r in s.strip().splitlines()
                    ]
                )

            A, B = to_mtx(A_in), to_mtx(B_in)

            st.subheader("Bimatrix Game")
            display_matrix = []
            for i in range(A.shape[0]):
                row = []
                for j in range(A.shape[1]):
                    row.append(f"({A[i,j]}, {B[i,j]})")
                display_matrix.append(row)

            df_display = pd.DataFrame(
                display_matrix,
                columns=[f"Strategy {j+1}" for j in range(A.shape[1])],
                index=[f"P1 Strategy {i+1}" for i in range(A.shape[0])],
            )
            st.dataframe(df_display)
            st.caption("Format: (Player 1 payoff, Player 2 payoff)")

            # Try to find Nash equilibria
            try:
                nash_game = nash.Game(A, B)
                eq = list(nash_game.support_enumeration())
                if eq:
                    st.success("**Nash Equilibria Found:**")
                    for idx, (p, q) in enumerate(eq):
                        st.write(f"Equilibrium {idx+1}:")
                        st.write(f"  Player 1 strategy: {p}")
                        st.write(f"  Player 2 strategy: {q}")
                else:
                    st.info(
                        "No mixed-strategy equilibrium found by support enumeration."
                    )
            except:
                st.info("Nash equilibrium computation not available.")

        except Exception as e:
            st.error(f"Error: {e}")


# ── Network Analysis ───────────────────────────────────────────────────────────
def network_analysis():
    st.title(t("network_an"))
    st.markdown("Find shortest paths in weighted graphs using Dijkstra's algorithm.")

    edges_in = st.text_area(
        "Edges (source, target, weight):", "A,B,1\nB,C,2\nA,C,4\nB,D,5\nC,D,1"
    )
    c1, c2 = st.columns(2)
    start = c1.text_input("Start node", "A")
    end = c2.text_input("End node", "D")

    if st.button(t("calculate_button")):
        try:
            G = nx.Graph()
            edges = []
            for line in edges_in.strip().splitlines():
                u, v, w = [x.strip() for x in line.split(",")]
                G.add_edge(u, v, weight=float(w))
                edges.append((u, v, float(w)))

            st.subheader("Graph Edges")
            st.dataframe(pd.DataFrame(edges, columns=["Source", "Target", "Weight"]))

            if start in G.nodes and end in G.nodes:
                path = nx.shortest_path(G, start, end, weight="weight")
                dist = nx.shortest_path_length(G, start, end, weight="weight")

                st.subheader("Shortest Path Analysis")
                c1, c2 = st.columns(2)
                c1.metric("Shortest Path", " → ".join(path))
                c2.metric("Total Distance", f"{dist:.2f}")

                # Show path details
                path_details = []
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    weight = G[u][v]["weight"]
                    path_details.append({"From": u, "To": v, "Weight": weight})

                if path_details:
                    st.subheader("Path Details")
                    st.dataframe(pd.DataFrame(path_details))
            else:
                st.error("Start or end node not found in graph.")

        except Exception as e:
            st.error(f"Error: {e}")


# ── Simulation (M/M/1) ────────────────────────────────────────────────────────
def simulation():
    st.title(t("simulation"))
    st.markdown(
        "Analyze M/M/1 queuing systems with Poisson arrivals and exponential service times."
    )

    st.header("M/M/1 Queue Analysis")
    c1, c2 = st.columns(2)
    λ = c1.number_input("Arrival rate λ (customers/hour)", 0.1, value=5.0, step=0.1)
    μ = c2.number_input("Service rate μ (customers/hour)", 0.1, value=6.0, step=0.1)

    if st.button(t("calculate_button")):
        if λ >= μ:
            st.error("❌ System is unstable (λ ≥ μ). Queue will grow indefinitely.")
        else:
            ρ = λ / μ
            L = λ / (μ - λ)
            Lq = λ**2 / (μ * (μ - λ))
            W = 1 / (μ - λ)
            Wq = λ / (μ * (μ - λ))

            st.subheader("Queue Performance Metrics")

            metrics_data = {
                "Metric": [
                    "Server Utilization (ρ)",
                    "Avg customers in system (L)",
                    "Avg customers in queue (Lq)",
                    "Avg time in system (W)",
                    "Avg waiting time (Wq)",
                ],
                "Value": [
                    f"{ρ:.4f}",
                    f"{L:.4f}",
                    f"{Lq:.4f}",
                    f"{W:.4f} hours",
                    f"{Wq:.4f} hours",
                ],
                "Formula": ["λ/μ", "λ/(μ-λ)", "λ²/[μ(μ-λ)]", "1/(μ-λ)", "λ/[μ(μ-λ)]"],
            }

            st.dataframe(pd.DataFrame(metrics_data))

            # Utilization warning
            if ρ > 0.8:
                st.warning(
                    "⚠️ High utilization (>80%). System may experience long delays."
                )
            elif ρ > 0.9:
                st.error(
                    "⚠️ Very high utilization (>90%). Consider increasing service capacity."
                )


# ── Project Management (basic PERT/CPM) ───────────────────────────────────────
def project_management():
    st.title(t("proj_man"))
    st.markdown("Basic PERT/CPM analysis for project scheduling.")

    data_in = st.text_area(
        "Tasks (Task, Duration, Dependencies):",
        "A,3,\nB,4,A\nC,2,A\nD,5,B\nE,2,C,D",
        height=120,
    )

    if st.button(t("calculate_button")):
        try:
            tasks = {}
            for line in data_in.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                name, dur = parts[0], int(parts[1])
                deps = [d for d in parts[2:] if d]
                tasks[name] = dict(duration=dur, dependencies=deps)

            st.subheader("Project Tasks")
            task_df = pd.DataFrame(
                [
                    {
                        "Task": name,
                        "Duration": info["duration"],
                        "Dependencies": (
                            ", ".join(info["dependencies"])
                            if info["dependencies"]
                            else "None"
                        ),
                    }
                    for name, info in tasks.items()
                ]
            )
            st.dataframe(task_df)

            # Simple forward pass calculation
            st.subheader("Schedule Analysis")
            earliest_start = {}
            earliest_finish = {}

            # Calculate earliest start and finish times
            def calculate_early_times(task):
                if task in earliest_start:
                    return earliest_start[task], earliest_finish[task]

                deps = tasks[task]["dependencies"]
                if not deps:
                    earliest_start[task] = 0
                else:
                    max_finish = 0
                    for dep in deps:
                        _, dep_finish = calculate_early_times(dep)
                        max_finish = max(max_finish, dep_finish)
                    earliest_start[task] = max_finish

                earliest_finish[task] = earliest_start[task] + tasks[task]["duration"]
                return earliest_start[task], earliest_finish[task]

            for task in tasks:
                calculate_early_times(task)

            schedule_df = pd.DataFrame(
                [
                    {
                        "Task": task,
                        "Duration": tasks[task]["duration"],
                        "Earliest Start": earliest_start[task],
                        "Earliest Finish": earliest_finish[task],
                    }
                    for task in tasks
                ]
            )
            st.dataframe(schedule_df)

            project_duration = max(earliest_finish.values())
            st.success(f"**Project Duration: {project_duration} time units**")

            st.info(
                "Note: This is a simplified PERT analysis. Critical path and slack calculations not implemented."
            )

        except Exception as e:
            st.error(f"Error: {e}")


# ── Inventory Control (EOQ) ────────────────────────────────────────────────────
def inventory_control():
    st.title(t("inv_ctrl"))
    st.markdown("Economic Order Quantity (EOQ) model for inventory optimization.")

    st.header("EOQ Parameters")
    D = st.number_input("Annual demand (D)", min_value=1, value=1000, step=1)
    K = st.number_input("Ordering cost per order (K)", min_value=1, value=50, step=1)
    h = st.number_input(
        "Holding cost per unit per year (h)", min_value=0.1, value=5.0, step=0.1
    )

    if st.button(t("calculate_button")):
        eoq = np.sqrt(2 * D * K / h)
        total_cost = np.sqrt(2 * D * K * h)
        order_frequency = D / eoq
        time_between_orders = 365 / order_frequency

        st.header("EOQ Results")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Economic Order Quantity (EOQ)", f"{eoq:.0f} units")
            st.metric("Total Annual Cost", f"${total_cost:.2f}")

        with col2:
            st.metric("Order Frequency", f"{order_frequency:.1f} orders/year")
            st.metric("Time Between Orders", f"{time_between_orders:.1f} days")

        # Cost breakdown
        st.subheader("Cost Analysis")
        ordering_cost = (D / eoq) * K
        holding_cost = (eoq / 2) * h

        cost_data = pd.DataFrame(
            {
                "Cost Component": ["Ordering Cost", "Holding Cost", "Total Cost"],
                "Annual Cost": [
                    f"${ordering_cost:.2f}",
                    f"${holding_cost:.2f}",
                    f"${total_cost:.2f}",
                ],
                "Formula": ["(D/Q)×K", "(Q/2)×h", "√(2×D×K×h)"],
            }
        )
        st.dataframe(cost_data)

        st.info(f"**EOQ Formula:** Q* = √(2DK/h) = √(2×{D}×{K}/{h}) = {eoq:.0f} units")


# ═══════════════════════════════════════════════════════════════════════════════
#                                 NAVIGATION
# ═══════════════════════════════════════════════════════════════════════════════
page_map = {
    t("homepage"): show_homepage,
    t("math_opt"): mathematical_optimization,
    t("assignment"): assignment_problem,
    t("transportation"): transportation_problem,
    t("sequencing"): sequencing_problem,
    t("game_theory"): game_theory,
    t("network_an"): network_analysis,
    t("simulation"): simulation,
    t("proj_man"): project_management,
    t("inv_ctrl"): inventory_control,
}

page_map[selection]()
