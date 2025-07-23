import streamlit as st
import pandas as pd
import numpy as np
import pulp
import networkx as nx
import plotly.graph_objects as go
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_iris
from scipy.optimize import linear_sum_assignment
import nashpy as nash

# --- Page Configuration ---
st.set_page_config(
    page_title="Enhanced Operational Research Toolkit",
    layout="wide"
)

# --- Language & Translation Setup ---
LANGUAGES = {
    "English": {
        "app_title": "Enhanced Operational Research Toolkit",
        "nav_title": "OR Toolkit Navigation",
        "nav_header": "Choose a module",
        "homepage": "Homepage",
        "math_opt": "Mathematical Optimization",
        "assignment": "Assignment Problem",
        "transportation": "Transportation Problem",
        "sequencing": "Sequencing Problem",
        "game_theory": "Game Theory",
        "network_an": "Network Analysis",
        "simulation": "Simulation",
        "proj_man": "Project Management",
        "inv_ctrl": "Inventory Control",
        "welcome": "Welcome to the Enhanced Operational Research (OR) Toolkit",
        "welcome_desc": "This comprehensive application includes advanced OR techniques with step-by-step solutions.",
        "modules_included": "Modules Included:",
        "select_module": "Select a module from the sidebar to get started.",
        "solve_button": "Solve",
        "calculate_button": "Calculate",
        "run_button": "Run",
        "generate_button": "Generate",
        # Mathematical Optimization
        "lp_solver_desc": "Solve Linear Programming problems with complete Simplex iterations and flexible constraint operators.",
        "problem_type": "Problem Type",
        "maximize": "Maximize",
        "minimize": "Minimize",
        "obj_coeffs": "Objective Function Coefficients (comma-separated)",
        "num_constraints": "Number of Constraints",
        "constraint_coeffs": "Coefficients for Constraint",
        "constraint_rhs": "RHS for Constraint",
        "step1_formulation": "Step 1: LP Model Formulation",
        "step2_feasible_region": "Step 2: Feasible Region Visualization",
        "step3_simplex_steps": "Step 3: Complete Simplex Method Iterations",
        "step4_results": "Step 4: Optimal Solution",
        "step5_conclusion": "Step 5: Conclusion & Interpretation",
        # Assignment Problem
        "assignment_desc": "Solve assignment problems using the Hungarian Algorithm.",
        "cost_matrix_input": "Enter Cost Matrix (comma-separated rows):",
        "hungarian_steps": "Hungarian Algorithm Steps",
        # Transportation Problem
        "transport_desc": "Solve transportation problems with supply and demand constraints.",
        "supply_input": "Supply (comma-separated):",
        "demand_input": "Demand (comma-separated):",
        "cost_matrix": "Cost Matrix (rows for sources, columns for destinations):",
        # Sequencing Problem
        "sequencing_desc": "Solve job sequencing using Johnson's Rule for 2-machine problems.",
        "job_times": "Enter job processing times (Job, Machine1, Machine2):",
        # Game Theory
        "game_theory_desc": "Analyze 2-player bimatrix games and find Nash equilibria.",
        "player1_matrix": "Player 1 Payoff Matrix:",
        "player2_matrix": "Player 2 Payoff Matrix:",
    },
    "العربية": {
        "app_title": "مجموعة أدوات بحوث العمليات المحسنة",
        "nav_title": "أدوات بحوث العمليات",
        "nav_header": "اختر أداة",
        "homepage": "الصفحة الرئيسية",
        "math_opt": "البرمجة الرياضية",
        "assignment": "مسألة التخصيص",
        "transportation": "مسألة النقل",
        "sequencing": "مسألة التسلسل",
        "game_theory": "نظرية الألعاب",
        "network_an": "تحليل الشبكات",
        "simulation": "المحاكاة",
        "proj_man": "إدارة المشاريع",
        "inv_ctrl": "مراقبة المخزون",
        "welcome": "أهلاً بك في مجموعة أدوات بحوث العمليات المحسنة",
        "welcome_desc": "هذا التطبيق الشامل يتضمن تقنيات OR المتقدمة مع حلول خطوة بخطوة.",
        "modules_included": "الأدوات المتضمنة:",
        "select_module": "اختر أداة من الشريط الجانبي للبدء.",
        "solve_button": "حل",
        "calculate_button": "احسب",
        "run_button": "تشغيل",
        "generate_button": "إنشاء",
        # البرمجة الرياضية
        "lp_solver_desc": "حل مسائل البرمجة الخطية مع جميع تكرارات السمبلكس ومشغلات القيود المرنة.",
        "problem_type": "نوع المسألة",
        "maximize": "تعظيم",
        "minimize": "تقليل",
        "obj_coeffs": "معاملات دالة الهدف (مفصولة بفواصل)",
        "num_constraints": "عدد القيود",
        "constraint_coeffs": "معاملات القيد رقم",
        "constraint_rhs": "الطرف الأيمن للقيد رقم",
        "step1_formulation": "الخطوة ١: صياغة نموذج البرمجة الخطية",
        "step2_feasible_region": "الخطوة ٢: رسم المنطقة المسموحة",
        "step3_simplex_steps": "الخطوة ٣: جميع تكرارات طريقة السمبلكس",
        "step4_results": "الخطوة ٤: الحل الأمثل",
        "step5_conclusion": "الخطوة ٥: الخلاصة والتفسير",
        # مسألة التخصيص
        "assignment_desc": "حل مسائل التخصيص باستخدام خوارزمية الهنغاري.",
        "cost_matrix_input": "أدخل مصفوفة التكلفة (صفوف مفصولة بفواصل):",
        "hungarian_steps": "خطوات خوارزمية الهنغاري",
        # مسألة النقل
        "transport_desc": "حل مسائل النقل مع قيود العرض والطلب.",
        "supply_input": "العرض (مفصول بفواصل):",
        "demand_input": "الطلب (مفصول بفواصل):",
        "cost_matrix": "مصفوفة التكلفة (صفوف للمصادر، أعمدة للوجهات):",
        # مسألة التسلسل
        "sequencing_desc": "حل تسلسل المهام باستخدام قاعدة جونسون للآلتين.",
        "job_times": "أدخل أوقات معالجة المهام (المهمة، الآلة١، الآلة٢):",
        # نظرية الألعاب
        "game_theory_desc": "تحليل ألعاب المصفوفة المزدوجة وإيجاد توازن ناش.",
        "player1_matrix": "مصفوفة عوائد اللاعب الأول:",
        "player2_matrix": "مصفوفة عوائد اللاعب الثاني:",
    }
}

# --- Language Selection & Helper ---
if 'lang' not in st.session_state:
    st.session_state.lang = "English"

def set_lang(lang):
    st.session_state.lang = lang

def t(key):
    return LANGUAGES[st.session_state.lang].get(key, key)

# Apply RTL layout for Arabic
if st.session_state.lang == "العربية":
    st.markdown("""
        <style>
        body, .stApp, .stButton, .stTextInput, .stTextArea, .stSelectbox, .stNumberInput {
            direction: rtl;
        }
        .stButton>button {
            width: 100%;
        }
        </style>
    """, unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.title(t("nav_title"))

selected_lang = st.sidebar.selectbox("Language / اللغة", options=["English", "العربية"], index=["English", "العربية"].index(st.session_state.lang))
if selected_lang != st.session_state.lang:
    set_lang(selected_lang)
    st.experimental_rerun()

selection = st.sidebar.radio(
    t("nav_header"),
    [t("homepage"), t("math_opt"), t("assignment"), t("transportation"), 
     t("sequencing"), t("game_theory"), t("network_an"), t("simulation"), 
     t("proj_man"), t("inv_ctrl")]
)

# --- Enhanced Simplex Class for Full Iterations ---
class SimplexSolver:
    def __init__(self, c, A, b, maximize=True):
        """
        Initialize Simplex solver
        c: coefficients of objective function
        A: constraint matrix
        b: RHS values
        maximize: True for maximization, False for minimization
        """
        self.c_original = np.array(c)
        self.A_original = np.array(A)
        self.b_original = np.array(b)
        self.maximize = maximize
        self.iterations = []
        self.basic_vars = []
        
    def solve(self):
        """Solve using Simplex method and return all iterations"""
        # Convert to standard form
        if not self.maximize:
            c = -self.c_original
        else:
            c = self.c_original.copy()
            
        A = self.A_original.copy()
        b = self.b_original.copy()
        
        # Add slack variables
        m, n = A.shape
        
        # Create initial tableau
        tableau = np.zeros((m + 1, n + m + 1))
        
        # Fill constraint rows
        tableau[:m, :n] = A
        tableau[:m, n:n+m] = np.eye(m)  # Identity matrix for slack variables
        tableau[:m, -1] = b
        
        # Fill objective row
        tableau[m, :n] = -c if self.maximize else c
        tableau[m, -1] = 0
        
        # Basic variables (slack variables initially)
        basic_vars = list(range(n, n + m))
        
        # Store initial tableau
        self.iterations.append({
            'iteration': 0,
            'tableau': tableau.copy(),
            'basic_vars': basic_vars.copy(),
            'entering_var': None,
            'leaving_var': None,
            'pivot_element': None
        })
        
        iteration = 0
        while True:
            iteration += 1
            
            # Check optimality (for maximization: all coefficients in obj row >= 0)
            obj_row = tableau[m, :-1]
            if self.maximize:
                if np.all(obj_row >= -1e-10):
                    break
                entering_col = np.argmin(obj_row)
            else:
                if np.all(obj_row <= 1e-10):
                    break
                entering_col = np.argmax(obj_row)
            
            # Find leaving variable (minimum ratio test)
            ratios = []
            for i in range(m):
                if tableau[i, entering_col] > 1e-10:
                    ratios.append(tableau[i, -1] / tableau[i, entering_col])
                else:
                    ratios.append(float('inf'))
            
            if all(r == float('inf') for r in ratios):
                # Unbounded solution
                break
                
            leaving_row = np.argmin(ratios)
            pivot_element = tableau[leaving_row, entering_col]
            
            # Store iteration info
            self.iterations.append({
                'iteration': iteration,
                'tableau': tableau.copy(),
                'basic_vars': basic_vars.copy(),
                'entering_var': entering_col,
                'leaving_var': basic_vars[leaving_row],
                'pivot_element': pivot_element,
                'ratios': ratios.copy()
            })
            
            # Pivot operation
            # Make pivot element 1
            tableau[leaving_row] = tableau[leaving_row] / pivot_element
            
            # Make other elements in pivot column 0
            for i in range(m + 1):
                if i != leaving_row:
                    tableau[i] = tableau[i] - tableau[i, entering_col] * tableau[leaving_row]
            
            # Update basic variables
            basic_vars[leaving_row] = entering_col
        
        # Store final tableau
        self.iterations.append({
            'iteration': iteration,
            'tableau': tableau.copy(),
            'basic_vars': basic_vars.copy(),
            'entering_var': None,
            'leaving_var': None,
            'pivot_element': None,
            'final': True
        })
        
        return self.iterations

# --- Enhanced 2D Plotting Function with Mixed Operators ---
def plot_feasible_region_2d_with_operators(constraints_data, obj_coeffs, is_maximize, optimal_solution):
    """Enhanced plot for 2-variable problems with mixed ≤ and ≥ constraints"""
    
    if len(obj_coeffs) != 2 or len(constraints_data) > 4:
        return None, None
    
    fig = go.Figure()
    
    # Create coordinate ranges
    x_max, y_max = 15, 15
    x_range = np.linspace(0, x_max, 1000)
    y_range = np.linspace(0, y_max, 1000)
    X, Y = np.meshgrid(x_range, y_range)
    
    # Start with the entire positive quadrant as feasible
    feasible_region = np.ones_like(X, dtype=bool)
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    corner_points = [(0, 0)]  # Origin might be feasible
    
    # Plot each constraint
    for i, constraint in enumerate(constraints_data):
        if len(constraint['coeffs']) >= 2:
            a, b, rhs = constraint['coeffs'][0], constraint['coeffs'][1], constraint['rhs']
            operator = constraint['operator']
            
            # Apply constraint to feasible region
            if operator == "≤":
                constraint_satisfied = (a * X + b * Y <= rhs)
            else:  # "≥"
                constraint_satisfied = (a * X + b * Y >= rhs)
            
            feasible_region &= constraint_satisfied
            
            # Plot constraint line: ax + by = rhs
            if b != 0:
                y_line = (rhs - a * x_range) / b
                valid_indices = (y_line >= 0) & (y_line <= y_max)
                
                line_style = dict(color=colors[i % len(colors)], width=3)
                if operator == "≥":
                    line_style['dash'] = 'dot'  # Distinguish ≥ constraints with dotted lines
                
                fig.add_trace(go.Scatter(
                    x=x_range[valid_indices],
                    y=y_line[valid_indices],
                    mode='lines',
                    name=f'Constraint {i+1}: {a:.1f}x₁ + {b:.1f}x₂ {operator} {rhs}',
                    line=line_style
                ))
            elif a != 0:
                x_line = rhs / a
                if 0 <= x_line <= x_max:
                    line_style = dict(color=colors[i % len(colors)], width=3)
                    if operator == "≥":
                        line_style['dash'] = 'dot'
                    
                    fig.add_trace(go.Scatter(
                        x=[x_line, x_line],
                        y=[0, y_max],
                        mode='lines',
                        name=f'Constraint {i+1}: {a:.1f}x₁ {operator} {rhs}',
                        line=line_style
                    ))
            
            # Find axis intersections
            if a != 0:
                x_intercept = rhs / a
                if x_intercept >= 0:
                    corner_points.append((x_intercept, 0))
            
            if b != 0:
                y_intercept = rhs / b
                if y_intercept >= 0:
                    corner_points.append((0, y_intercept))
    
    # Find constraint intersections
    for i in range(len(constraints_data)):
        for j in range(i+1, len(constraints_data)):
            c1, c2 = constraints_data[i], constraints_data[j]
            if len(c1['coeffs']) >= 2 and len(c2['coeffs']) >= 2:
                # Solve system: a1*x + b1*y = rhs1, a2*x + b2*y = rhs2
                A_matrix = np.array([
                    [c1['coeffs'][0], c1['coeffs'][1]],
                    [c2['coeffs'][0], c2['coeffs'][1]]
                ])
                b_vector = np.array([c1['rhs'], c2['rhs']])
                
                try:
                    intersection = np.linalg.solve(A_matrix, b_vector)
                    x_int, y_int = intersection[0], intersection[1]
                    
                    if x_int >= -0.001 and y_int >= -0.001:
                        # Check if point satisfies all constraints
                        feasible = True
                        for k, constraint in enumerate(constraints_data):
                            if len(constraint['coeffs']) >= 2:
                                lhs_value = constraint['coeffs'][0] * x_int + constraint['coeffs'][1] * y_int
                                if constraint['operator'] == "≤":
                                    if lhs_value > constraint['rhs'] + 0.001:
                                        feasible = False
                                        break
                                else:  # "≥"
                                    if lhs_value < constraint['rhs'] - 0.001:
                                        feasible = False
                                        break
                        
                        if feasible:
                            corner_points.append((max(0, x_int), max(0, y_int)))
                except np.linalg.LinAlgError:
                    pass  # Lines are parallel
    
    # Remove duplicate points and sort
    corner_points = list(set([(round(p[0], 6), round(p[1], 6)) for p in corner_points]))
    corner_points = [p for p in corner_points if p[0] >= 0 and p[1] >= 0]
    
    # Filter corner points to keep only feasible ones
    feasible_corners = []
    for point in corner_points:
        x1, x2 = point[0], point[1]
        feasible = True
        for constraint in constraints_data:
            if len(constraint['coeffs']) >= 2:
                lhs_value = constraint['coeffs'][0] * x1 + constraint['coeffs'][1] * x2
                if constraint['operator'] == "≤":
                    if lhs_value > constraint['rhs'] + 0.001:
                        feasible = False
                        break
                else:  # "≥"
                    if lhs_value < constraint['rhs'] - 0.001:
                        feasible = False
                        break
        if feasible:
            feasible_corners.append(point)
    
    corner_points = feasible_corners
    
    if len(corner_points) > 2:
        # Sort points to form proper polygon
        centroid_x = sum(p[0] for p in corner_points) / len(corner_points)
        centroid_y = sum(p[1] for p in corner_points) / len(corner_points)
        
        def angle_from_centroid(point):
            return np.arctan2(point[1] - centroid_y, point[0] - centroid_x)
        
        corner_points.sort(key=angle_from_centroid)
        
        # Plot feasible region
        x_coords = [p[0] for p in corner_points] + [corner_points[0][0]]
        y_coords = [p[1] for p in corner_points] + [corner_points[0][1]]
        
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            fill='toself',
            fillcolor='rgba(0, 255, 0, 0.3)',
            line=dict(color='rgba(0, 255, 0, 0.8)', width=2),
            name='Feasible Region',
            hoverinfo='skip'
        ))
    
    # Plot corner points
    if corner_points:
        fig.add_trace(go.Scatter(
            x=[p[0] for p in corner_points],
            y=[p[1] for p in corner_points],
            mode='markers+text',
            marker=dict(size=10, color='black', symbol='circle'),
            text=[f'({p[0]:.1f}, {p[1]:.1f})' for p in corner_points],
            textposition='top center',
            name='Corner Points',
            textfont=dict(size=10, color='black')
        ))
    
    # Plot optimal solution
    if optimal_solution and len(optimal_solution) >= 2:
        fig.add_trace(go.Scatter(
            x=[optimal_solution[0]],
            y=[optimal_solution[1]],
            mode='markers+text',
            marker=dict(size=15, color='red', symbol='star'),
            text=f'Optimal: ({optimal_solution[0]:.2f}, {optimal_solution[1]:.2f})',
            textposition='top center',
            name='Optimal Solution',
            textfont=dict(size=12, color='red', family='Arial Black')
        ))
        
        # Plot objective function line through optimal point
        if len(obj_coeffs) >= 2 and obj_coeffs[1] != 0:
            optimal_obj_value = obj_coeffs[0] * optimal_solution[0] + obj_coeffs[1] * optimal_solution[1]
            
            if obj_coeffs[1] != 0:
                x_obj_range = np.linspace(0, x_max, 100)
                y_obj_line = (optimal_obj_value - obj_coeffs[0] * x_obj_range) / obj_coeffs[1]
                valid_obj = (y_obj_line >= 0) & (y_obj_line <= y_max)
                
                if np.any(valid_obj):
                    fig.add_trace(go.Scatter(
                        x=x_obj_range[valid_obj],
                        y=y_obj_line[valid_obj],
                        mode='lines',
                        line=dict(color='red', dash='dash', width=4),
                        name=f'Objective Line: {obj_coeffs[0]}x₁ + {obj_coeffs[1]}x₂ = {optimal_obj_value:.2f}',
                        hoverinfo='skip'
                    ))
    
    # Customize layout
    fig.update_layout(
        title=dict(
            text='Feasible Region with Mixed Constraints (Solid: ≤, Dotted: ≥)',
            x=0.5,
            font=dict(size=16, family='Arial', color='darkblue')
        ),
        xaxis=dict(
            title='x₁ (Variable 1)',
            range=[0, x_max],
            gridcolor='lightgray',
            showgrid=True
        ),
        yaxis=dict(
            title='x₂ (Variable 2)',
            range=[0, y_max],
            gridcolor='lightgray',
            showgrid=True
        ),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="Black",
            borderwidth=1
        ),
        plot_bgcolor='white',
        width=800,
        height=600
    )
    
    return fig, corner_points

# --- Module Functions ---
def show_homepage():
    st.title(t("welcome"))
    st.markdown(t("welcome_desc"))
    st.header(t("modules_included"))
    st.markdown(f"""
    - **{t("math_opt")}:** Complete Simplex method with all iterations, 2D feasible region plots, and flexible constraint operators (≤, ≥)
    - **{t("assignment")}:** Hungarian Algorithm for assignment optimization
    - **{t("transportation")}:** Transportation problem solver with VAM
    - **{t("sequencing")}:** Johnson's Rule for 2-machine sequencing
    - **{t("game_theory")}:** Bimatrix games and Nash equilibria
    - **{t("network_an")}:** Network analysis and shortest paths
    - **{t("simulation")}:** Queuing theory and Monte Carlo simulation
    - **{t("proj_man")}:** PERT/CPM for project management
    - **{t("inv_ctrl")}:** Economic Order Quantity (EOQ)
    """)
    st.info(t("select_module"))

def mathematical_optimization():
    st.title(t("math_opt"))
    st.markdown(t("lp_solver_desc"))

    # Input section
    problem_type = st.radio(t("problem_type"), (t("maximize"), t("minimize")))
    objective_coeffs = st.text_input(t("obj_coeffs"), "3, 5")
    num_constraints = st.number_input(t("num_constraints"), min_value=1, max_value=6, value=2, step=1)

    constraints = []
    for i in range(num_constraints):
        st.subheader(f"{t('constraint_coeffs')} #{i+1}")
        cols = st.columns(3)  # Changed to 3 columns to accommodate operator selection
        
        constraint_coeffs = cols[0].text_input(f"Coefficients #{i+1}", 
                                             value="2, 1" if i == 0 else "1, 3", key=f"c{i}")
        
        # Add operator selection
        constraint_operator = cols[1].selectbox(f"Operator #{i+1}", 
                                              options=["≤", "≥"], 
                                              index=0, key=f"op{i}")
        
        constraint_rhs = cols[2].number_input(f"RHS #{i+1}", 
                                            value=20 if i == 0 else 30, key=f"rhs{i}")
        
        constraints.append({
            'coeffs': constraint_coeffs, 
            'operator': constraint_operator,
            'rhs': constraint_rhs
        })

    if st.button(t("solve_button")):
        with st.spinner("Solving..."):
            try:
                # Parse inputs
                obj_coeffs = [float(c.strip()) for c in objective_coeffs.split(',')]
                constraints_data = []
                
                for i, c in enumerate(constraints):
                    coeffs = [float(cf.strip()) for cf in c['coeffs'].split(',')]
                    constraints_data.append({
                        'coeffs': coeffs, 
                        'operator': c['operator'],
                        'rhs': c['rhs']
                    })
                
                # Step 1: Model Formulation
                st.header(t("step1_formulation"))
                
                obj_type = "Maximize" if problem_type == t("maximize") else "Minimize"
                st.latex(f"\\text{{{obj_type}}} \\quad Z = " + " + ".join([f"{coef:.1f}x_{i+1}" for i, coef in enumerate(obj_coeffs)]))
                
                st.write("**Subject to:**")
                for i, constraint in enumerate(constraints_data):
                    constraint_str = " + ".join([f"{coef:.1f}x_{j+1}" for j, coef in enumerate(constraint['coeffs'])])
                    operator_latex = "\\leq" if constraint['operator'] == "≤" else "\\geq"
                    st.latex(f"{constraint_str} {operator_latex} {constraint['rhs']}")
                
                st.latex("x_i \\geq 0 \\quad \\forall i")

                # Convert all constraints to standard form (≤) for PuLP
                standard_constraints_data = []
                for constraint in constraints_data:
                    if constraint['operator'] == "≤":
                        standard_constraints_data.append(constraint)
                    else:  # "≥" constraint - convert to "≤" by multiplying by -1
                        standard_constraints_data.append({
                            'coeffs': [-coef for coef in constraint['coeffs']],
                            'operator': "≤",
                            'rhs': -constraint['rhs']
                        })

                # Solve with PuLP
                prob_type = pulp.LpMaximize if problem_type == t("maximize") else pulp.LpMinimize
                prob = pulp.LpProblem("LP_Problem", prob_type)
                
                num_vars = len(obj_coeffs)
                variables = [pulp.LpVariable(f'x{j+1}', lowBound=0) for j in range(num_vars)]
                
                # Objective function
                prob += pulp.lpSum([obj_coeffs[j] * variables[j] for j in range(num_vars)])

                # Add constraints (using standard form for PuLP)
                for i, constraint in enumerate(standard_constraints_data):
                    prob += pulp.lpSum([constraint['coeffs'][j] * variables[j] for j in range(num_vars)]) <= constraint['rhs'], f"Constraint_{i+1}"
                
                prob.solve(pulp.PULP_CBC_CMD(msg=0))
                
                # Get optimal solution for plotting
                optimal_solution = None
                if prob.status == pulp.LpStatusOptimal:
                    optimal_solution = [v.varValue for v in prob.variables()]

                # Step 2: Feasible Region Plot (for 2-variable problems)
                if len(obj_coeffs) == 2 and len(constraints_data) >= 2:
                    st.header(t("step2_feasible_region"))
                    
                    # Use original constraints_data (not standard form) for plotting
                    fig, corner_points = plot_feasible_region_2d_with_operators(
                        constraints_data, obj_coeffs, 
                        problem_type == t("maximize"), optimal_solution
                    )
                    
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Corner points evaluation table
                        st.subheader("Corner Points Analysis")
                        
                        if corner_points and len(corner_points) > 0:
                            corner_analysis = []
                            
                            for point in corner_points:
                                x1, x2 = point[0], point[1]
                                
                                # Check if point satisfies all original constraints
                                feasible = True
                                for constraint in constraints_data:
                                    if len(constraint['coeffs']) >= 2:
                                        lhs_value = constraint['coeffs'][0] * x1 + constraint['coeffs'][1] * x2
                                        if constraint['operator'] == "≤":
                                            if lhs_value > constraint['rhs'] + 0.001:
                                                feasible = False
                                                break
                                        else:  # "≥"
                                            if lhs_value < constraint['rhs'] - 0.001:
                                                feasible = False
                                                break
                                
                                if feasible:
                                    obj_value = obj_coeffs[0] * x1 + obj_coeffs[1] * x2
                                    corner_analysis.append({
                                        'Corner Point': f'({x1:.2f}, {x2:.2f})',
                                        'x₁': f'{x1:.2f}',
                                        'x₂': f'{x2:.2f}',
                                        'Objective Value': f'{obj_value:.2f}',
                                        'Feasible': 'Yes'
                                    })
                            
                            if corner_analysis:
                                corner_df = pd.DataFrame(corner_analysis)
                                st.dataframe(corner_df, use_container_width=True)
                                
                                # Highlight optimal corner point
                                if optimal_solution:
                                    optimal_obj = obj_coeffs[0] * optimal_solution[0] + obj_coeffs[1] * optimal_solution[1]
                                    optimal_text = f"The optimal corner point is ({optimal_solution[0]:.2f}, {optimal_solution[1]:.2f}) with objective value {optimal_obj:.2f}"
                                    
                                    if problem_type == t("maximize"):
                                        st.success(f"🎯 **Maximum Found:** {optimal_text}")
                                    else:
                                        st.success(f"🎯 **Minimum Found:** {optimal_text}")
                        
                        # Constraints verification
                        st.subheader("Constraint Analysis at Optimal Solution")
                        if optimal_solution and len(optimal_solution) >= 2:
                            constraint_analysis = []
                            for i, constraint in enumerate(constraints_data):
                                if len(constraint['coeffs']) >= 2:
                                    lhs_value = constraint['coeffs'][0] * optimal_solution[0] + constraint['coeffs'][1] * optimal_solution[1]
                                    
                                    if constraint['operator'] == "≤":
                                        slack = constraint['rhs'] - lhs_value
                                        binding = "Yes" if abs(slack) < 0.001 else "No"
                                        constraint_status = f"≤ {constraint['rhs']:.3f} (Slack: {slack:.3f})"
                                    else:  # "≥"
                                        surplus = lhs_value - constraint['rhs']
                                        binding = "Yes" if abs(surplus) < 0.001 else "No"
                                        constraint_status = f"≥ {constraint['rhs']:.3f} (Surplus: {surplus:.3f})"
                                    
                                    constraint_analysis.append({
                                        'Constraint': f'Constraint {i+1}',
                                        'LHS Value': f'{lhs_value:.3f}',
                                        'Status': constraint_status,
                                        'Binding': binding
                                    })
                            
                            constraint_df = pd.DataFrame(constraint_analysis)
                            st.dataframe(constraint_df, use_container_width=True)
                            
                            st.info("💡 **Note:** Binding constraints determine the optimal solution. Slack (for ≤) is unused capacity, Surplus (for ≥) is excess above minimum requirement.")

                # Step 3: Complete Simplex Method (using standard form)
                st.header(t("step3_simplex_steps"))
                
                # Show conversion to standard form if there are ≥ constraints
                has_ge_constraints = any(c['operator'] == "≥" for c in constraints_data)
                if has_ge_constraints:
                    st.subheader("Conversion to Standard Form")
                    st.write("**Original Constraints → Standard Form (≤):**")
                    
                    for i, (orig, std) in enumerate(zip(constraints_data, standard_constraints_data)):
                        if orig['operator'] == "≥":
                            orig_str = " + ".join([f"{coef:.1f}x_{j+1}" for j, coef in enumerate(orig['coeffs'])])
                            std_str = " + ".join([f"{coef:.1f}x_{j+1}" for j, coef in enumerate(std['coeffs'])])
                            st.write(f"**Constraint {i+1}:** {orig_str} ≥ {orig['rhs']} → {std_str} ≤ {std['rhs']}")
                        else:
                            constraint_str = " + ".join([f"{coef:.1f}x_{j+1}" for j, coef in enumerate(orig['coeffs'])])
                            st.write(f"**Constraint {i+1}:** {constraint_str} ≤ {orig['rhs']} (unchanged)")
                    
                    st.divider()
                
                # Prepare data for SimplexSolver using standard form
                A = np.array([constraint['coeffs'] for constraint in standard_constraints_data])
                b = np.array([constraint['rhs'] for constraint in standard_constraints_data])
                c = np.array(obj_coeffs)
                
                # Solve using custom Simplex solver
                solver = SimplexSolver(c, A, b, maximize=(problem_type == t("maximize")))
                iterations = solver.solve()
                
                # Display all iterations
                for iter_data in iterations:
                    iteration_num = iter_data['iteration']
                    tableau = iter_data['tableau']
                    basic_vars = iter_data['basic_vars']
                    
                    if iteration_num == 0:
                        st.subheader("Initial Simplex Tableau")
                    elif iter_data.get('final'):
                        st.subheader("Final Optimal Tableau")
                    else:
                        st.subheader(f"Iteration {iteration_num}")
                        
                        if iter_data.get('entering_var') is not None:
                            st.write(f"**Entering Variable:** x{iter_data['entering_var'] + 1}")
                        if iter_data.get('leaving_var') is not None:
                            leaving_var_name = f"x{iter_data['leaving_var'] + 1}" if iter_data['leaving_var'] < len(obj_coeffs) else f"s{iter_data['leaving_var'] - len(obj_coeffs) + 1}"
                            st.write(f"**Leaving Variable:** {leaving_var_name}")
                        if iter_data.get('pivot_element') is not None:
                            st.write(f"**Pivot Element:** {iter_data['pivot_element']:.4f}")
                    
                    # Create tableau display
                    m, n = tableau.shape
                    num_original_vars = len(obj_coeffs)
                    
                    # Column headers
                    headers = ['Basic Var'] + [f'x{i+1}' for i in range(num_original_vars)] + \
                             [f's{i+1}' for i in range(m-1)] + ['RHS']
                    
                    # Row headers (basic variables)
                    basic_var_names = []
                    for var in basic_vars:
                        if var < num_original_vars:
                            basic_var_names.append(f'x{var + 1}')
                        else:
                            basic_var_names.append(f's{var - num_original_vars + 1}')
                    basic_var_names.append('Z')
                    
                    # Create DataFrame for display
                    display_tableau = tableau.copy()
                    tableau_df = pd.DataFrame(display_tableau, columns=headers[1:])
                    tableau_df.insert(0, 'Basic Var', basic_var_names)
                    
                    st.dataframe(tableau_df.round(4), use_container_width=True)
                    
                    # Show ratios for non-final iterations
                    if iter_data.get('ratios') and not iter_data.get('final'):
                        ratios_display = []
                        for i, ratio in enumerate(iter_data['ratios']):
                            if ratio != float('inf'):
                                ratios_display.append(f"Row {i+1}: {ratio:.4f}")
                            else:
                                ratios_display.append(f"Row {i+1}: ∞")
                        st.write("**Minimum Ratio Test:**", " | ".join(ratios_display))
                    
                    st.divider()

                # Step 4: Results
                st.header(t("step4_results"))
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="Status", value=pulp.LpStatus[prob.status])
                with col2:
                    if prob.status == pulp.LpStatusOptimal:
                        st.metric(label="Optimal Value", value=f"{pulp.value(prob.objective):,.4f}")
                
                # Variables table
                if prob.status == pulp.LpStatusOptimal:
                    results_df = pd.DataFrame([
                        {"Variable": v.name, "Optimal Value": f"{v.varValue:.4f}"} 
                        for v in prob.variables()
                    ])
                    st.dataframe(results_df, use_container_width=True)

                # Step 5: Conclusion
                st.header(t("step5_conclusion"))
                
                if prob.status == pulp.LpStatusOptimal:
                    st.success("✅ **Optimal Solution Found!**")
                    
                    optimal_vars = {v.name: v.varValue for v in prob.variables()}
                    optimal_obj = pulp.value(prob.objective)
                    
                    interpretation = f"""
                    **Solution Summary:**
                    
                    **Decision Variables:**
                    """
                    for var_name, var_value in optimal_vars.items():
                        interpretation += f"\n- {var_name} = {var_value:.4f}"
                    
                    interpretation += f"""
                    
                    **Optimal Objective Value:** {optimal_obj:.4f}
                    
                    **Constraint Types Summary:**
                    """
                    
                    le_count = sum(1 for c in constraints_data if c['operator'] == "≤")
                    ge_count = sum(1 for c in constraints_data if c['operator'] == "≥")
                    interpretation += f"\n- ≤ constraints: {le_count}"
                    interpretation += f"\n- ≥ constraints: {ge_count}"
                    
                    interpretation += f"""
                    
                    **Solution Method:** 
                    The Simplex Method required {len(iterations)-2} iterations to reach optimality.
                    {'Constraints with ≥ operators were converted to standard ≤ form for solution.' if has_ge_constraints else 'All constraints were already in standard ≤ form.'}
                    
                    **Business Interpretation:**
                    - ≤ constraints represent capacity limitations or resource availability
                    - ≥ constraints represent minimum requirements or demand satisfaction
                    This solution optimizes the objective while respecting both limitations and requirements.
                    """
                    
                    st.markdown(interpretation)
                    
                else:
                    st.error(f"❌ **Problem Status:** {pulp.LpStatus[prob.status]}")

            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.error("Please check your input format. Ensure coefficients are comma-separated numbers.")

def assignment_problem():
    st.title(t("assignment"))
    st.markdown(t("assignment_desc"))
    
    st.subheader("Hungarian Algorithm for Assignment Problem")
    
    # Default cost matrix
    default_matrix = "9, 2, 7, 8\n6, 4, 3, 7\n5, 8, 1, 8\n7, 6, 9, 4"
    
    matrix_input = st.text_area(t("cost_matrix_input"), default_matrix, height=100)
    
    if st.button(t("solve_button")):
        try:
            # Parse cost matrix
            lines = matrix_input.strip().split('\n')
            cost_matrix = []
            for line in lines:
                row = [float(x.strip()) for x in line.split(',')]
                cost_matrix.append(row)
            
            cost_matrix = np.array(cost_matrix)
            st.write("**Input Cost Matrix:**")
            st.dataframe(pd.DataFrame(cost_matrix))
            
            # Solve using scipy's implementation of Hungarian algorithm
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            total_cost = cost_matrix[row_indices, col_indices].sum()
            
            st.header(t("hungarian_steps"))
            
            # Step 1: Show original matrix
            st.subheader("Step 1: Original Cost Matrix")
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(cost_matrix, cmap='Blues')
            
            # Add text annotations
            for i in range(cost_matrix.shape[0]):
                for j in range(cost_matrix.shape[1]):
                    text = ax.text(j, i, f'{cost_matrix[i, j]:.0f}',
                                 ha="center", va="center", color="black", fontweight='bold')
            
            ax.set_xticks(np.arange(cost_matrix.shape[1]))
            ax.set_yticks(np.arange(cost_matrix.shape[0]))
            ax.set_xticklabels([f'Task {j+1}' for j in range(cost_matrix.shape[1])])
            ax.set_yticklabels([f'Worker {i+1}' for i in range(cost_matrix.shape[0])])
            ax.set_title("Original Cost Matrix")
            plt.colorbar(im)
            st.pyplot(fig)
            
            # Step 2: Row reduction
            st.subheader("Step 2: Row Reduction")
            row_reduced = cost_matrix - cost_matrix.min(axis=1, keepdims=True)
            st.write("Subtract minimum of each row:")
            st.dataframe(pd.DataFrame(row_reduced))
            
            # Step 3: Column reduction
            st.subheader("Step 3: Column Reduction")
            col_reduced = row_reduced - row_reduced.min(axis=0, keepdims=True)
            st.write("Subtract minimum of each column:")
            st.dataframe(pd.DataFrame(col_reduced))
            
            # Step 4: Final assignment
            st.subheader("Step 4: Optimal Assignment")
            
            assignment_matrix = np.zeros_like(cost_matrix)
            assignment_matrix[row_indices, col_indices] = 1
            
            # Create visualization of assignment
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(cost_matrix, cmap='Blues', alpha=0.3)
            
            # Highlight assigned cells
            for i, j in zip(row_indices, col_indices):
                rect = plt.Rectangle((j-0.4, i-0.4), 0.8, 0.8, fill=False, 
                                   edgecolor='red', linewidth=3)
                ax.add_patch(rect)
            
            # Add text annotations
            for i in range(cost_matrix.shape[0]):
                for j in range(cost_matrix.shape[1]):
                    color = 'red' if assignment_matrix[i, j] == 1 else 'black'
                    weight = 'bold' if assignment_matrix[i, j] == 1 else 'normal'
                    text = ax.text(j, i, f'{cost_matrix[i, j]:.0f}',
                                 ha="center", va="center", color=color, fontweight=weight)
            
            ax.set_xticks(np.arange(cost_matrix.shape[1]))
            ax.set_yticks(np.arange(cost_matrix.shape[0]))
            ax.set_xticklabels([f'Task {j+1}' for j in range(cost_matrix.shape[1])])
            ax.set_yticklabels([f'Worker {i+1}' for i in range(cost_matrix.shape[0])])
            ax.set_title("Optimal Assignment (Red boxes show assignments)")
            st.pyplot(fig)
            
            # Results
            st.subheader("Assignment Results")
            results = []
            for i, j in zip(row_indices, col_indices):
                results.append({
                    'Worker': f'Worker {i+1}',
                    'Task': f'Task {j+1}',
                    'Cost': cost_matrix[i, j]
                })
            
            results_df = pd.DataFrame(results)
            st.dataframe(results_df)
            
            st.success(f"**Total Minimum Cost: {total_cost:.2f}**")
            
        except Exception as e:
            st.error(f"Error: {e}. Please check your input format.")

def transportation_problem():
    st.title(t("transportation"))
    st.markdown(t("transport_desc"))
    
    # Input section
    supply_input = st.text_input(t("supply_input"), "300, 400, 500")
    demand_input = st.text_input(t("demand_input"), "250, 350, 400, 200")
    
    st.subheader(t("cost_matrix"))
    cost_matrix_input = st.text_area("Cost per unit from each source to each destination:",
                                   "19, 30, 50, 10\n70, 30, 40, 60\n40, 8, 70, 20", height=100)
    
    if st.button(t("solve_button")):
        try:
            # Parse inputs
            supply = [int(x.strip()) for x in supply_input.split(',')]
            demand = [int(x.strip()) for x in demand_input.split(',')]
            
            lines = cost_matrix_input.strip().split('\n')
            costs = []
            for line in lines:
                row = [int(x.strip()) for x in line.split(',')]
                costs.append(row)
            costs = np.array(costs)
            
            # Check if balanced
            total_supply = sum(supply)
            total_demand = sum(demand)
            
            st.header("Problem Analysis")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Supply", total_supply)
            col2.metric("Total Demand", total_demand)
            col3.metric("Balance", "Balanced" if total_supply == total_demand else "Unbalanced")
            
            # Display input data
            st.subheader("Step 1: Problem Data")
            
            # Supply and demand table
            supply_demand_data = {
                'Source': [f'S{i+1}' for i in range(len(supply))] + ['Demand'],
                'Supply/Demand': supply + ['-']
            }
            for j in range(len(demand)):
                supply_demand_data[f'D{j+1}'] = [costs[i][j] for i in range(len(supply))] + [demand[j]]
            
            st.dataframe(pd.DataFrame(supply_demand_data))
            
            # Solve using Vogel's Approximation Method (VAM)
            st.subheader("Step 2: Vogel's Approximation Method (VAM)")
            
            # Initialize
            supply_remaining = supply.copy()
            demand_remaining = demand.copy()
            allocation = np.zeros((len(supply), len(demand)))
            cost_matrix = costs.copy()
            
            iteration = 0
            vam_steps = []
            
            while max(supply_remaining) > 0 and max(demand_remaining) > 0:
                iteration += 1
                
                # Calculate penalties for rows
                row_penalties = []
                for i in range(len(supply)):
                    if supply_remaining[i] > 0:
                        available_costs = []
                        for j in range(len(demand)):
                            if demand_remaining[j] > 0:
                                available_costs.append(cost_matrix[i][j])
                        if len(available_costs) >= 2:
                            available_costs.sort()
                            penalty = available_costs[1] - available_costs[0]
                        else:
                            penalty = 0
                        row_penalties.append(penalty)
                    else:
                        row_penalties.append(-1)
                
                # Calculate penalties for columns
                col_penalties = []
                for j in range(len(demand)):
                    if demand_remaining[j] > 0:
                        available_costs = []
                        for i in range(len(supply)):
                            if supply_remaining[i] > 0:
                                available_costs.append(cost_matrix[i][j])
                        if len(available_costs) >= 2:
                            available_costs.sort()
                            penalty = available_costs[1] - available_costs[0]
                        else:
                            penalty = 0
                        col_penalties.append(penalty)
                    else:
                        col_penalties.append(-1)
                
                # Find maximum penalty
                max_row_penalty = max([p for p in row_penalties if p >= 0], default=0)
                max_col_penalty = max([p for p in col_penalties if p >= 0], default=0)
                
                if max_row_penalty >= max_col_penalty:
                    # Select row with maximum penalty
                    selected_row = row_penalties.index(max_row_penalty)
                    # Find minimum cost in this row
                    min_cost = float('inf')
                    selected_col = -1
                    for j in range(len(demand)):
                        if demand_remaining[j] > 0 and cost_matrix[selected_row][j] < min_cost:
                            min_cost = cost_matrix[selected_row][j]
                            selected_col = j
                else:
                    # Select column with maximum penalty
                    selected_col = col_penalties.index(max_col_penalty)
                    # Find minimum cost in this column
                    min_cost = float('inf')
                    selected_row = -1
                    for i in range(len(supply)):
                        if supply_remaining[i] > 0 and cost_matrix[i][selected_col] < min_cost:
                            min_cost = cost_matrix[i][selected_col]
                            selected_row = i
                
                # Allocate
                allocated = min(supply_remaining[selected_row], demand_remaining[selected_col])
                allocation[selected_row][selected_col] = allocated
                supply_remaining[selected_row] -= allocated
                demand_remaining[selected_col] -= allocated
                
                vam_steps.append({
                    'iteration': iteration,
                    'row_penalties': row_penalties.copy(),
                    'col_penalties': col_penalties.copy(),
                    'selected_cell': (selected_row, selected_col),
                    'allocated': allocated,
                    'cost': min_cost
                })
                
                if iteration > 20:  # Safety break
                    break
            
            # Display VAM iterations
            for step in vam_steps:
                with st.expander(f"VAM Iteration {step['iteration']}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Row Penalties:**", [f"{p:.0f}" if p >= 0 else "X" for p in step['row_penalties']])
                    with col2:
                        st.write("**Column Penalties:**", [f"{p:.0f}" if p >= 0 else "X" for p in step['col_penalties']])
                    
                    st.write(f"**Selected Cell:** S{step['selected_cell'][0]+1} → D{step['selected_cell'][1]+1}")
                    st.write(f"**Allocated:** {step['allocated']} units")
                    st.write(f"**Unit Cost:** {step['cost']}")
            
            # Final allocation
            st.subheader("Step 3: Final Allocation")
            allocation_df = pd.DataFrame(allocation, 
                                       columns=[f'D{j+1}' for j in range(len(demand))],
                                       index=[f'S{i+1}' for i in range(len(supply))])
            st.dataframe(allocation_df)
            
            # Calculate total cost
            total_cost = np.sum(allocation * costs)
            st.success(f"**Minimum Transportation Cost: {total_cost:.2f}**")
            
            # Summary
            st.subheader("Step 4: Solution Summary")
            non_zero_allocations = []
            for i in range(len(supply)):
                for j in range(len(demand)):
                    if allocation[i][j] > 0:
                        non_zero_allocations.append({
                            'Route': f'S{i+1} → D{j+1}',
                            'Quantity': allocation[i][j],
                            'Unit Cost': costs[i][j],
                            'Total Cost': allocation[i][j] * costs[i][j]
                        })
            
            summary_df = pd.DataFrame(non_zero_allocations)
            st.dataframe(summary_df)
            
        except Exception as e:
            st.error(f"Error: {e}. Please check your input format.")

def sequencing_problem():
    st.title(t("sequencing"))
    st.markdown(t("sequencing_desc"))
    
    st.subheader("Johnson's Rule for 2-Machine Sequencing")
    
    # Default job data
    default_jobs = "A, 5, 2\nB, 1, 6\nC, 9, 7\nD, 3, 8\nE, 10, 4"
    
    job_input = st.text_area(t("job_times"), default_jobs, height=120)
    
    if st.button(t("solve_button")):
        try:
            # Parse job data
            lines = job_input.strip().split('\n')
            jobs = []
            for line in lines:
                parts = [x.strip() for x in line.split(',')]
                job_name = parts[0]
                machine1_time = float(parts[1])
                machine2_time = float(parts[2])
                jobs.append({
                    'name': job_name,
                    'machine1': machine1_time,
                    'machine2': machine2_time
                })
            
            st.header("Johnson's Rule Solution")
            
            # Step 1: Display input data
            st.subheader("Step 1: Job Processing Times")
            job_df = pd.DataFrame(jobs)
            st.dataframe(job_df)
            
            # Step 2: Apply Johnson's Rule
            st.subheader("Step 2: Apply Johnson's Rule")
            
            remaining_jobs = jobs.copy()
            first_sequence = []
            last_sequence = []
            step_details = []
            
            step = 0
            while remaining_jobs:
                step += 1
                
                # Find minimum processing time
                min_time = float('inf')
                selected_job = None
                machine = None
                
                for job in remaining_jobs:
                    if job['machine1'] < min_time:
                        min_time = job['machine1']
                        selected_job = job
                        machine = 1
                    if job['machine2'] < min_time:
                        min_time = job['machine2']
                        selected_job = job
                        machine = 2
                
                # Add to appropriate sequence
                if machine == 1:
                    first_sequence.append(selected_job)
                    position = "first"
                else:
                    last_sequence.insert(0, selected_job)
                    position = "last"
                
                step_details.append({
                    'step': step,
                    'job': selected_job['name'],
                    'min_time': min_time,
                    'machine': f'Machine {machine}',
                    'position': position,
                    'remaining': [j['name'] for j in remaining_jobs if j != selected_job]
                })
                
                remaining_jobs.remove(selected_job)
            
            # Display steps
            for detail in step_details:
                with st.expander(f"Step {detail['step']}: Select Job {detail['job']}"):
                    st.write(f"**Minimum time:** {detail['min_time']} on {detail['machine']}")
                    st.write(f"**Position:** Place {detail['position']}")
                    st.write(f"**Remaining jobs:** {', '.join(detail['remaining'])}")
            
            # Final sequence
            optimal_sequence = first_sequence + last_sequence
            st.subheader("Step 3: Optimal Job Sequence")
            
            sequence_names = [job['name'] for job in optimal_sequence]
            st.success(f"**Optimal Sequence:** {' → '.join(sequence_names)}")
            
            # Step 4: Calculate completion times and create Gantt chart
            st.subheader("Step 4: Gantt Chart and Performance Metrics")
            
            # Calculate completion times
            machine1_schedule = []
            machine2_schedule = []
            
            current_time_m1 = 0
            current_time_m2 = 0
            
            for job in optimal_sequence:
                # Machine 1
                start_time_m1 = current_time_m1
                finish_time_m1 = start_time_m1 + job['machine1']
                machine1_schedule.append({
                    'job': job['name'],
                    'start': start_time_m1,
                    'finish': finish_time_m1,
                    'duration': job['machine1']
                })
                current_time_m1 = finish_time_m1
                
                # Machine 2
                start_time_m2 = max(current_time_m2, finish_time_m1)
                finish_time_m2 = start_time_m2 + job['machine2']
                machine2_schedule.append({
                    'job': job['name'],
                    'start': start_time_m2,
                    'finish': finish_time_m2,
                    'duration': job['machine2']
                })
                current_time_m2 = finish_time_m2
            
            # Create Gantt chart data
            gantt_data = []
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
            
            for i, task in enumerate(machine1_schedule):
                gantt_data.append(dict(
                    Task=f"Machine 1",
                    Start=task['start'],
                    Finish=task['finish'],
                    Resource=task['job'],
                    Description=f"Job {task['job']}"
                ))
            
            for i, task in enumerate(machine2_schedule):
                gantt_data.append(dict(
                    Task=f"Machine 2",
                    Start=task['start'],
                    Finish=task['finish'],
                    Resource=task['job'],
                    Description=f"Job {task['job']}"
                ))
            
            # Create Gantt chart
            fig = ff.create_gantt(gantt_data, colors=colors, index_col='Resource',
                                show_colorbar=True, group_tasks=True, 
                                title='Optimal Job Sequence - Gantt Chart')
            st.plotly_chart(fig, use_container_width=True)
            
            # Performance metrics
            st.subheader("Step 5: Performance Metrics")
            
            total_elapsed_time = max(machine1_schedule[-1]['finish'], machine2_schedule[-1]['finish'])
            machine1_idle_time = total_elapsed_time - sum(task['duration'] for task in machine1_schedule)
            machine2_idle_time = total_elapsed_time - sum(task['duration'] for task in machine2_schedule)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Elapsed Time", f"{total_elapsed_time} time units")
            col2.metric("Machine 1 Idle Time", f"{machine1_idle_time} time units")
            col3.metric("Machine 2 Idle Time", f"{machine2_idle_time} time units")
            
            # Detailed schedule
            st.subheader("Detailed Schedule")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Machine 1 Schedule:**")
                m1_df = pd.DataFrame(machine1_schedule)
                st.dataframe(m1_df)
            
            with col2:
                st.write("**Machine 2 Schedule:**")
                m2_df = pd.DataFrame(machine2_schedule)
                st.dataframe(m2_df)
                
        except Exception as e:
            st.error(f"Error: {e}. Please check your input format.")

def game_theory():
    st.title(t("game_theory"))
    st.markdown(t("game_theory_desc"))
    
    st.subheader("2-Player Bimatrix Game Analysis")
    
    # Default payoff matrices (Prisoner's Dilemma)
    default_p1 = "3, 0\n5, 1"
    default_p2 = "3, 5\n0, 1"
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Player 1 (Row Player) Payoff Matrix:**")
        p1_input = st.text_area(t("player1_matrix"), default_p1, height=80, key="p1")
    
    with col2:
        st.write("**Player 2 (Column Player) Payoff Matrix:**")
        p2_input = st.text_area(t("player2_matrix"), default_p2, height=80, key="p2")
    
    if st.button(t("solve_button")):
        try:
            # Parse payoff matrices
            def parse_matrix(matrix_str):
                lines = matrix_str.strip().split('\n')
                matrix = []
                for line in lines:
                    row = [float(x.strip()) for x in line.split(',')]
                    matrix.append(row)
                return np.array(matrix)
            
            A = parse_matrix(p1_input)  # Player 1's payoff matrix
            B = parse_matrix(p2_input)  # Player 2's payoff matrix
            
            st.header("Game Analysis")
            
            # Step 1: Display the game
            st.subheader("Step 1: Game Representation")
            
            # Create bimatrix display
            rows, cols = A.shape
            bimatrix_display = []
            
            for i in range(rows):
                row = []
                for j in range(cols):
                    cell = f"({A[i,j]:.1f}, {B[i,j]:.1f})"
                    row.append(cell)
                bimatrix_display.append(row)
            
            bimatrix_df = pd.DataFrame(bimatrix_display, 
                                     columns=[f'Strategy {j+1}' for j in range(cols)],
                                     index=[f'Strategy {i+1}' for i in range(rows)])
            bimatrix_df.index.name = 'Player 1 \\ Player 2'
            
            st.dataframe(bimatrix_df)
            st.caption("Format: (Player 1 payoff, Player 2 payoff)")
            
            # Step 2: Check for dominant strategies
            st.subheader("Step 2: Dominance Analysis")
            
            # Check for dominant strategies for Player 1
            st.write("**Player 1 Dominant Strategies:**")
            p1_dominant = []
            for i in range(rows):
                for k in range(rows):
                    if i != k:
                        if np.all(A[i, :] > A[k, :]):
                            p1_dominant.append(f"Strategy {i+1} strictly dominates Strategy {k+1}")
                        elif np.all(A[i, :] >= A[k, :]) and np.any(A[i, :] > A[k, :]):
                            p1_dominant.append(f"Strategy {i+1} weakly dominates Strategy {k+1}")
            
            if p1_dominant:
                for dom in p1_dominant:
                    st.write(f"- {dom}")
            else:
                st.write("- No dominant strategies found for Player 1")
            
            # Check for dominant strategies for Player 2
            st.write("**Player 2 Dominant Strategies:**")
            p2_dominant = []
            for j in range(cols):
                for k in range(cols):
                    if j != k:
                        if np.all(B[:, j] > B[:, k]):
                            p2_dominant.append(f"Strategy {j+1} strictly dominates Strategy {k+1}")
                        elif np.all(B[:, j] >= B[:, k]) and np.any(B[:, j] > B[:, k]):
                            p2_dominant.append(f"Strategy {j+1} weakly dominates Strategy {k+1}")
            
            if p2_dominant:
                for dom in p2_dominant:
                    st.write(f"- {dom}")
            else:
                st.write("- No dominant strategies found for Player 2")
            
            # Step 3: Find Nash equilibria using nashpy
            st.subheader("Step 3: Nash Equilibrium Analysis")
            
            try:
                game = nash.Game(A, B)
                
                # Find all Nash equilibria
                equilibria = list(game.support_enumeration())
                
                if equilibria:
                    st.write(f"**Found {len(equilibria)} Nash Equilibrium/Equilibria:**")
                    
                    for idx, (s1, s2) in enumerate(equilibria):
                        st.write(f"\n**Equilibrium {idx + 1}:**")
                        
                        # Player 1 strategy
                        p1_strategy = []
                        for i, prob in enumerate(s1):
                            if prob > 1e-10:
                                p1_strategy.append(f"Strategy {i+1}: {prob:.4f}")
                        
                        # Player 2 strategy
                        p2_strategy = []
                        for j, prob in enumerate(s2):
                            if prob > 1e-10:
                                p2_strategy.append(f"Strategy {j+1}: {prob:.4f}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Player 1:**")
                            for strategy in p1_strategy:
                                st.write(f"- {strategy}")
                        
                        with col2:
                            st.write("**Player 2:**")
                            for strategy in p2_strategy:
                                st.write(f"- {strategy}")
                        
                        # Calculate expected payoffs
                        expected_payoff_1 = float(s1.T @ A @ s2)
                        expected_payoff_2 = float(s1.T @ B @ s2)
                        
                        st.write(f"**Expected Payoffs:** Player 1: {expected_payoff_1:.4f}, Player 2: {expected_payoff_2:.4f}")
                        
                        # Check if pure strategy
                        is_pure_1 = np.sum(s1 == 1) == 1 and np.sum(s1 == 0) == len(s1) - 1
                        is_pure_2 = np.sum(s2 == 1) == 1 and np.sum(s2 == 0) == len(s2) - 1
                        
                        if is_pure_1 and is_pure_2:
                            st.info("This is a **Pure Strategy Nash Equilibrium**")
                        else:
                            st.info("This is a **Mixed Strategy Nash Equilibrium**")
                
                else:
                    st.warning("No Nash equilibria found using support enumeration.")
                
            except Exception as e:
                st.error(f"Error in Nash equilibrium calculation: {e}")
            
            # Step 4: Game classification
            st.subheader("Step 4: Game Classification")
            
            # Check if zero-sum
            is_zero_sum = np.allclose(A, -B)
            if is_zero_sum:
                st.success("✅ This is a **Zero-Sum Game**")
            else:
                st.info("ℹ️ This is a **Non-Zero-Sum Game**")
            
            # Check for common game types
            if A.shape == (2, 2):
                # Check for Prisoner's Dilemma
                if (A[1,0] > A[0,0] and A[1,0] > A[1,1] and A[0,1] > A[0,0] and A[0,1] > A[1,1] and
                    B[0,1] > B[0,0] and B[0,1] > B[1,1] and B[1,0] > B[0,0] and B[1,0] > B[1,1] and
                    A[0,0] + A[1,1] > A[0,1] + A[1,0] and B[0,0] + B[1,1] > B[0,1] + B[1,0]):
                    st.warning("⚠️ This appears to be a **Prisoner's Dilemma**")
                
                # Check for coordination game
                elif A[0,0] > A[0,1] and A[0,0] > A[1,0] and A[1,1] > A[1,0] and A[1,1] > A[0,1]:
                    st.info("🤝 This appears to be a **Coordination Game**")
            
            # Step 5: Strategy recommendations
            st.subheader("Step 5: Strategic Analysis and Recommendations")
            
            analysis = """
            **Game Theory Insights:**
            
            1. **Nash Equilibrium**: The solution concept where no player can unilaterally deviate and improve their payoff.
            
            2. **Dominant Strategies**: If a player has a dominant strategy, they should always play it regardless of opponent's choice.
            
            3. **Mixed Strategies**: When no pure strategy equilibrium exists, players randomize according to equilibrium probabilities.
            
            **Strategic Recommendations:**
            - If dominant strategies exist, play them
            - In mixed strategy equilibria, randomize according to the calculated probabilities
            - Consider the opponent's rational behavior when making decisions
            """
            
            st.markdown(analysis)
            
        except Exception as e:
            st.error(f"Error: {e}. Please check your input format.")

# Network Analysis (simplified version)
def network_analysis():
    st.title("Network Analysis")
    st.markdown("Define a network with edges and weights to find the shortest path between nodes.")

    edge_data = st.text_area("Enter edges (source, target, weight), one per line:", "A,B,1\nB,C,2\nA,C,4\nB,D,5\nC,D,1")
    cols = st.columns(2)
    start_node = cols[0].text_input("Start Node", "A")
    end_node = cols[1].text_input("End Node", "D")

    if st.button(t("calculate_button")):
        with st.spinner("Analyzing network..."):
            try:
                G = nx.Graph()
                lines = edge_data.strip().split('\n')
                for line in lines:
                    source, target, weight = line.split(',')
                    G.add_edge(source.strip(), target.strip(), weight=float(weight))
                
                path = nx.shortest_path(G, source=start_node, target=end_node, weight='weight')
                path_length = nx.shortest_path_length(G, source=start_node, target=end_node, weight='weight')
                
                st.metric(label="Shortest Path", value=' -> '.join(path))
                st.metric(label="Total Distance", value=f"{path_length:.2f}")

            except Exception as e:
                st.error(f"An error occurred: {e}")

def simulation():
    st.title("Simulation")
    st.header("Queuing Theory (M/M/1 Model)")
    
    cols = st.columns(2)
    arrival_rate = cols[0].number_input("Arrival Rate (λ)", min_value=0.1, value=5.0, step=0.1)
    service_rate = cols[1].number_input("Service Rate (μ)", min_value=0.1, value=6.0, step=0.1)
    
    if st.button(t("calculate_button")):
        if arrival_rate >= service_rate:
            st.error("Arrival rate must be less than service rate.")
        else:
            rho = arrival_rate / service_rate
            Lq = (arrival_rate ** 2) / (service_rate * (service_rate - arrival_rate))
            L = arrival_rate / (service_rate - arrival_rate)
            Wq = Lq / arrival_rate
            W = L / arrival_rate
            
            metrics = {
                "Metric": ["Server Utilization (ρ)", "Avg Customers in Queue (Lq)", "Avg Customers in System (L)", "Avg Wait Time in Queue (Wq)", "Avg Wait Time in System (W)"],
                "Value": [f"{rho:.4f}", f"{Lq:.4f}", f"{L:.4f}", f"{Wq:.4f}", f"{W:.4f}"]
            }
            st.table(pd.DataFrame(metrics))

def project_management():
    st.title("Project Management (PERT/CPM)")
    task_data_input = st.text_area("Enter tasks (Task, Duration, Dependencies):", "A,3,\nB,4,A\nC,2,A\nD,5,B\nE,2,C,D")

    if st.button(t("calculate_button")):
        try:
            lines = task_data_input.strip().split('\n')
            tasks, G = {}, nx.DiGraph()
            for line in lines:
                parts = [p.strip() for p in line.split(',')]
                task_id, duration, deps = parts[0], int(parts[1]), [d for d in parts[2:] if d]
                tasks[task_id] = {'duration': duration, 'dependencies': deps}
                G.add_node(task_id, duration=duration)
                for dep in deps: G.add_edge(dep, task_id)

            es, ef = {}, {}
            for node in nx.topological_sort(G):
                es[node] = max([ef.get(p, 0) for p in G.predecessors(node)], default=0)
                ef[node] = es[node] + tasks[node]['duration']

            project_duration = max(ef.values())
            lf, ls = {}, {}
            for node in reversed(list(nx.topological_sort(G))):
                lf[node] = min([ls.get(s, project_duration) for s in G.successors(node)], default=project_duration)
                ls[node] = lf[node] - tasks[node]['duration']

            critical_path = [node for node in G.nodes() if es[node] == ls[node]]
            
            st.metric("Project Duration", project_duration)
            st.metric("Critical Path", ' -> '.join(critical_path))
            
            results_df = pd.DataFrame({
                "Duration": [d['duration'] for d in tasks.values()], "ES": list(es.values()), "EF": list(ef.values()),
                "LS": list(ls.values()), "LF": list(lf.values()), "Slack": [ls[t] - es[t] for t in tasks]
            }, index=tasks.keys())
            st.dataframe(results_df)

        except Exception as e:
            st.error(f"Error: {e}")

def inventory_control():
    st.title("Inventory Control")
    st.header("Economic Order Quantity (EOQ)")

    annual_demand = st.number_input("Annual Demand (D)", min_value=1, value=1000)
    ordering_cost = st.number_input("Ordering Cost per Order (S)", min_value=1, value=50)
    holding_cost = st.number_input("Holding Cost per Unit per Year (H)", min_value=0.1, value=5.0, step=0.1)

    if st.button(t("calculate_button")):
        if annual_demand > 0 and ordering_cost > 0 and holding_cost > 0:
            eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost)
            total_cost_at_eoq = np.sqrt(2 * annual_demand * ordering_cost * holding_cost)

            st.metric("Economic Order Quantity (EOQ)", f"{eoq:,.2f} units")
            st.metric("Minimum Total Annual Cost", f"${total_cost_at_eoq:,.2f}")

            q = np.linspace(1, eoq * 2, 400)
            annual_holding_cost = (q / 2) * holding_cost
            annual_ordering_cost = (annual_demand / q) * ordering_cost
            total_cost_curve = annual_holding_cost + annual_ordering_cost
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=q, y=annual_holding_cost, mode='lines', name="Holding Cost"))
            fig.add_trace(go.Scatter(x=q, y=annual_ordering_cost, mode='lines', name="Ordering Cost"))
            fig.add_trace(go.Scatter(x=q, y=total_cost_curve, mode='lines', name="Total Cost", line=dict(width=4)))
            fig.add_vline(x=eoq, line_width=2, line_dash="dash", line_color="red", annotation_text=f"EOQ = {eoq:.2f}")
            
            fig.update_layout(title="Inventory Costs vs. Order Quantity", xaxis_title="Order Quantity (Q)", yaxis_title="Annual Cost ($)")
            st.plotly_chart(fig)
        else:
            st.error("All input values must be greater than zero.")

# --- Main App Logic ---
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
