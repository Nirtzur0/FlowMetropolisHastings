import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_schematic():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Styles
    box_props = dict(boxstyle='round,pad=0.5', facecolor='#e0f7fa', edgecolor='#006064', linewidth=2)
    flow_props = dict(boxstyle='round,pad=0.5', facecolor='#ffcdd2', edgecolor='#b71c1c', linewidth=2)
    local_props = dict(boxstyle='round,pad=0.5', facecolor='#c8e6c9', edgecolor='#1b5e20', linewidth=2)
    mh_props = dict(boxstyle='round,pad=0.5', facecolor='#ffe0b2', edgecolor='#e65100', linewidth=2)
    buffer_props = dict(boxstyle='square,pad=0.5', facecolor='#eeeeee', edgecolor='#424242', linestyle='--', linewidth=1.5)
    
    # Nodes
    
    # buffers
    ax.text(1.5, 5, "Training Buffer\nSamples", ha='center', va='center', bbox=buffer_props, fontsize=10)
    ax.arrow(1.5, 4.2, 0, -1.0, head_width=0.15, head_length=0.2, fc='k', ec='k', linestyle='--')
    ax.text(1.7, 3.6, "Train φ", fontsize=9)
    
    # State
    ax.text(6, 7, "Current State $x_t$", ha='center', va='center', bbox=box_props, fontsize=12, fontweight='bold')
    
    # Decision Node (Circle)
    circle = patches.Circle((6, 5), radius=0.6, facecolor='#bdbdbd', edgecolor='black', linewidth=1.5)
    ax.add_patch(circle)
    ax.text(6, 5, "Select\nKernel", ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Arrows from State to Decision
    ax.arrow(6, 6.4, 0, -0.6, head_width=0.2, head_length=0.2, fc='k', ec='k')
    
    # Flow Proposal (Left)
    ax.text(3, 3, "Flow Proposal\n$x' \\sim q_\\phi(x')$\n(ODE Integration)", ha='center', va='center', bbox=flow_props, fontsize=10)
    
    # Local Proposal (Right)
    ax.text(9, 3, "Local Proposal\n$x' \\sim K(x'|x_t)$", ha='center', va='center', bbox=local_props, fontsize=10)
    
    # Arrows from Decision
    # To Flow
    ax.annotate("", xy=(3, 3.6), xytext=(5.5, 4.8), arrowprops=dict(arrowstyle="->", lw=1.5, connectionstyle="arc3,rad=0.2"))
    ax.text(3.8, 4.5, "Global ($p_{glob}$)", fontsize=9, rotation=0)
    
    # To Local
    ax.annotate("", xy=(9, 3.6), xytext=(6.5, 4.8), arrowprops=dict(arrowstyle="->", lw=1.5, connectionstyle="arc3,rad=-0.2"))
    ax.text(7.5, 4.5, "Local ($1 - p_{glob}$)", fontsize=9, rotation=0)
    
    # MH Step
    ax.text(6, 1, "MH Acceptance\n$\\alpha(x_t, x')$", ha='center', va='center', bbox=mh_props, fontsize=12, fontweight='bold')
    
    # Arrows to MH
    ax.annotate("", xy=(5, 1.5), xytext=(3, 2.4), arrowprops=dict(arrowstyle="->", lw=1.5, connectionstyle="arc3,rad=0.1"))
    ax.annotate("", xy=(7, 1.5), xytext=(9, 2.4), arrowprops=dict(arrowstyle="->", lw=1.5, connectionstyle="arc3,rad=-0.1"))
    
    # Title
    ax.set_title("DiffMCMC Algorithm Schematic", fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig('paper/figures/schematic.png', dpi=300)
    print("Saved schematic to paper/figures/schematic.png")

if __name__ == "__main__":
    draw_schematic()
