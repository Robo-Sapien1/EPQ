import matplotlib.pyplot as plt
import numpy as np

def generate_dynamic_drone_graphs():
    """
    Generates 4 key graphs for EPQ, including Dynamic Dimension Sizing.
    """
    
    # --- 1. SETUP (2035 Era Tech) ---
    ranges_km = np.linspace(1, 200, 100) # Test up to 200km
    
    # Physics Constants
    struct_frac = 0.30        # 30% Structure
    energy_k = 0.0006         # Energy Cost (Cruise)
    power_min_frac = 0.073    # Power Cost (Hover)
    
    # Drone Definitions
    # 'base_span': The wing span needed for short-range urban hops
    fleet = {
        "Standard": {"pay": 11.9, "base_span": 1.6, "col": "blue"},   
        "Oversize": {"pay": 29.8, "base_span": 2.4, "col": "orange"},
        "H&B":      {"pay": 45.0, "base_span": 3.2, "col": "red"}
    }

    # Setup 2x2 Grid
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('2035 Drone Network: Sizing & Physics Model', fontsize=16)
    
    ax_weight = axs[0, 0]
    ax_span   = axs[0, 1]  # NEW: Dimensions Graph
    ax_batt   = axs[1, 0]
    ax_eff    = axs[1, 1]

    # --- 2. CALCULATIONS ---
    for name, data in fleet.items():
        weights = []
        spans = []
        batt_masses = []
        efficiencies = []
        
        # Calculate Base Weight (Minimum possible weight for this class)
        # Used to scale the wings
        denom_base = (1 - struct_frac) - power_min_frac
        w_base = data['pay'] / denom_base

        for r in ranges_km:
            # 1. Determine Weight (Power vs Energy Limited)
            denom_E = (1 - struct_frac) - (r * energy_k)
            denom_P = (1 - struct_frac) - power_min_frac
            effective_denom = min(denom_E, denom_P)
            
            if effective_denom <= 0:
                # Impossible flight
                weights.append(np.nan)
                spans.append(np.nan)
                batt_masses.append(np.nan)
                efficiencies.append(np.nan)
            else:
                w_total = data['pay'] / effective_denom
                weights.append(w_total)
                
                # 2. Dynamic Dimensions (Wing Span Scaling)
                # If weight increases by 10%, Wing Area needs to increase by 10%.
                # Since Area ~ Span^2, Span increases by sqrt(1.10)
                # Formula: New_Span = Base_Span * sqrt(Current_Weight / Base_Weight)
                scale_factor = np.sqrt(w_total / w_base)
                spans.append(data['base_span'] * scale_factor)

                # 3. Battery Stats
                w_batt = w_total - (w_total * struct_frac) - data['pay']
                batt_masses.append(w_batt)
                
                # 4. Efficiency (Payload as % of Total)
                efficiencies.append((data['pay'] / w_total) * 100)

        # Plot Lines
        ax_weight.plot(ranges_km, weights, label=name, color=data['col'], linewidth=2)
        ax_span.plot(ranges_km, spans, label=name, color=data['col'], linewidth=2, linestyle='--')
        ax_batt.plot(ranges_km, batt_masses, label=name, color=data['col'], linewidth=2)
        ax_eff.plot(ranges_km, efficiencies, label=name, color=data['col'], linewidth=2)

    # --- 3. FORMATTING ---
    
    # Graph A: Weight vs Range
    ax_weight.set_title('A. Total Takeoff Weight')
    ax_weight.set_ylabel('Weight (kg)')
    ax_weight.set_xlabel('Range (km)')
    ax_weight.grid(True, alpha=0.3)
    ax_weight.legend()
    ax_weight.axvspan(0, 50, color='green', alpha=0.1, label='Urban Zone')

    # Graph B: Wing Span (Dimensions) vs Range
    ax_span.set_title('B. Required Wing Span (Dynamic Sizing)')
    ax_span.set_ylabel('Wing Span (meters)')
    ax_span.set_xlabel('Range (km)')
    ax_span.grid(True, alpha=0.3)
    ax_span.text(10, 3.3, "Short Range = Constant Span", fontsize=9)
    ax_span.text(150, 4.0, "Long Range = Wider Wings", fontsize=9)

    # Graph C: Battery Weight
    ax_batt.set_title('C. Battery Mass Required')
    ax_batt.set_ylabel('Battery Weight (kg)')
    ax_batt.set_xlabel('Range (km)')
    ax_batt.grid(True, alpha=0.3)

    # Graph D: Efficiency
    ax_eff.set_title('D. Payload Efficiency (Payload / Total Weight)')
    ax_eff.set_ylabel('Efficiency (%)')
    ax_eff.set_xlabel('Range (km)')
    ax_eff.set_ylim(0, 70)
    ax_eff.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Run
generate_dynamic_drone_graphs()