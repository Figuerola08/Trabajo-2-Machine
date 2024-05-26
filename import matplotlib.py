import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def create_flowchart():
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.text(0.5, 0.9, 'Carga de Datos', ha='center', va='center', fontsize=12, bbox=dict(facecolor='lightblue', edgecolor='black', boxstyle='round,pad=0.5'))
    ax.text(0.5, 0.8, 'Preprocesamiento de Datos', ha='center', va='center', fontsize=12, bbox=dict(facecolor='lightgreen', edgecolor='black', boxstyle='round,pad=0.5'))
    ax.text(0.5, 0.7, 'Transformación de la Variable Objetivo', ha='center', va='center', fontsize=12, bbox=dict(facecolor='lightyellow', edgecolor='black', boxstyle='round,pad=0.5'))
    ax.text(0.5, 0.6, 'División del Dataset', ha='center', va='center', fontsize=12, bbox=dict(facecolor='lightcoral', edgecolor='black', boxstyle='round,pad=0.5'))
    ax.text(0.5, 0.5, 'Entrenamiento del Modelo', ha='center', va='center', fontsize=12, bbox=dict(facecolor='lightgray', edgecolor='black', boxstyle='round,pad=0.5'))
    ax.text(0.5, 0.4, 'Evaluación del Modelo', ha='center', va='center', fontsize=12, bbox=dict(facecolor='lightpink', edgecolor='black', boxstyle='round,pad=0.5'))
    ax.text(0.5, 0.3, 'Balanceo de Clases (para SVM)', ha='center', va='center', fontsize=12, bbox=dict(facecolor='lightcyan', edgecolor='black', boxstyle='round,pad=0.5'))
    ax.text(0.5, 0.2, 'Ajuste de Hiperparámetros', ha='center', va='center', fontsize=12, bbox=dict(facecolor='lightgoldenrodyellow', edgecolor='black', boxstyle='round,pad=0.5'))
    ax.text(0.5, 0.1, 'Predicción y Evaluación Final', ha='center', va='center', fontsize=12, bbox=dict(facecolor='lightsteelblue', edgecolor='black', boxstyle='round,pad=0.5'))

    for i in range(9, 1, -1):
        ax.arrow(0.5, i*0.1-0.05, 0, -0.05, head_width=0.02, head_length=0.02, fc='black', ec='black')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.title('Diagrama de Flujo de la Metodología')
    plt.show()

create_flowchart()
