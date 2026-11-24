from pathlib import Path 
import time 
from src.agent import graph, tools 

# 1. Carga del CSV y cálculo de pérdidas 
t0 = time.perf_counter() 
cleaned = tools.load_and_clean_csv("msft_stock.csv") 
losses = tools.calculate_loss_vector(cleaned) 
t_data = time.perf_counter() - t0 

# 2. Motor matemático determinista (POT) 
t0 = time.perf_counter() 
results = tools.apply_pot_method(losses, alpha=0.99, threshold=0.0331) 
t_quant = time.perf_counter() - t0 

# 3. Gráfica diagnóstica 
t0 = time.perf_counter() 
plot_path = tools.plot_tail_diagnostics( losses, 0.0331, results["xi"], results["beta"] ) 
t_plot = time.perf_counter() - t0 

# 4. Flujo completo con LangGraph y OpenAI (LLM) 
t0 = time.perf_counter() 
state = { 
    "messages": [{ 
        "role": "user", 
        "content": ( "Aplica POT con umbral 3.31% al 99% de confianza y genera gráfica." ), 
    }], 
    "df_path": str(Path("msft_stock.csv").resolve()),
} 
result = graph.build_graph().invoke(state)
t_graph_total = time.perf_counter() - t0 

print("\n--- DESGLOSE DE TIEMPOS DE RESPUESTA ---") 
print( f"1. Carga y cálculo de pérdidas (Data Agent): {t_data*1000:.2f} ms ({len(cleaned)} filas)" ) 
print(f"2. Optimización matemática POT (Quant Agent): {t_quant*1000:.2f} ms") 
print(f"3. Renderizado de gráfica Matplotlib: {t_plot*1000:.2f} ms") 
print( f"4. Ciclo completo de agentes (con LLM OpenAI): {t_graph_total:.2f} s\\n" )