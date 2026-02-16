"""
Evaluation:
    - Ce module calcule la fréquence d'apparition des attributs de la base de données 
    utilisé pour étude theorique sur les taux conditionnel.
    
    Class History_analyzer récupere les données et calcule les métriques:
        - taux global pour chaque valeurs attribut (attr / total);
        - le taux conditional pour l'attribut le plus fréquent;
        - fréquence d'apparition d'un valeur pour les messages classés <<communs>> et <<anomalie>>
    Sortie:
        - Données: analytical_data.md
        - Graphiques: analytical_plots.png
    
    - Hypotèse: Données une quantité assez large des donnés en isolant les n attributs les plus fréquents,
    on peut détecter si une message depasse le seuil d'anomalie (threshold). Ce détection est basée sur 
    l'évaluation conditionnelle entre les attributs observés et les attributs les plus fréquents.
    

"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import history as history

class History_analyzer:
    def __init__(self, fields, tag_c='tag'):
        self.fields = fields
        self.tag_c = tag_c
        self.message_input = {field: history.History() for field in fields}
        self.tagged_z = {field: history.History() for field in fields}
        self.tagged_one = {field: history.History() for field in fields}
        
        # statistics
        self.overal_stats = {}
        self.cond_stats = {}
        self.tagged_stats = {0: {}, 1: {}}
    
    def update_histograms(self, df):
        for _, row in df.iterrows():
            tag = row[self.tag_c]
            
            for field in self.fields:
                value = row[field]
                self.message_input[field].updat_h(value)
                if tag == 1:
                    self.tagged_one[field].updat_h(value)
                elif tag == 0:
                    self.tagged_z[field].updat_h(value)
        print(f"size of anomalie: {len(self.tagged_one)}")
    
    def calculate_rates(self, histogram):
        total = sum(histogram.values())
        if total == 0 : return {}
        return {key: count/total for key, count in histogram.items()}
    
    def get_max_per_attrb(self, attrb, hist_dict):
        max_value, max_rate, max_type = None, 0, None
        for field, hist in hist_dict.items():
            rates = self.calculate_rates(hist)
            if rates and field == attrb:
                if len(rates) > 0:
                    field_max_value, field_max_rate = max(rates.items(), key=lambda x: x[1])
                
                    if field_max_rate > max_rate:
                        max_type = field
                        max_rate = field_max_rate
                        max_value = field_max_value
        return max_type, max_value, max_rate
    
    def get_max_rate_attrb(self, hist_dict):
        max_value, max_rate, max_type = None, 0, None
        for field, hist in hist_dict.items():
            rates = self.calculate_rates(hist)
            if rates:
                if len(rates) > 0:
                    field_max_value, field_max_rate = max(rates.items(), key=lambda x: x[1])
                
                    if field_max_rate > max_rate:
                        max_type = field
                        max_rate = field_max_rate
                        max_value = field_max_value
        return max_type, max_value, max_rate
    
    def get_max_rate_attrb_list(self, hist_dict):
        max_list = {}
        for field, hist in hist_dict.items():
            rates = self.calculate_rates(hist)
            if rates:
                if len(rates) > 0:
                    field_max_value, field_max_rate = max(rates.items(), key=lambda x: x[1])
                    max_list[field] = (field_max_value, field_max_rate)
        return max_list
    
    def calculate_condt_avg(self, df, max_field, max_value):
        filtered_df = df[df[max_field]==max_value]
        cond_avg = {}
        for field in self.fields:
            if field != max_field and pd.api.types.is_numeric_dtype(df[field]):
                cond_avg[field] = filtered_df[field].mean()
        return self.calculate_rates(cond_avg)
    
    def generate_reports(self, df):
        """Generate statistical reports"""
        # Overall rates
        #print("\n" + "="*60)
        #print("OVERALL ATTRIBUTE RATES (Mixed Data)")
        #print("="*60)
        #for field in self.fields:
        #    rates = self.calculate_rates(self.message_input[field])
        #    print(f"\n{field}:")
        #    sorted_rates = sorted(rates.items(), key=lambda x: x[1], reverse=True)[:5]
        #    for value, rate in sorted_rates:
        #        print(f"  {value}: {rate:.4f} ({rate*100:.2f}%)")
        
        print("\n" + "="*60)
        print("OVERALL ATTRIBUTE RATES (Common behavior)")
        print("="*60)
        for field in self.fields:
            rates = self.calculate_rates(self.tagged_z[field])
            print(f"\n{field}:")
            sorted_rates = sorted(rates.items(), key=lambda x: x[1], reverse=True)[:5]
            for value, rate in sorted_rates:
                print(f"  {value}: {rate:.4f} ({rate*100:.2f}%)")
        
        print("\n" + "="*60)
        print("OVERALL ATTRIBUTE RATES (Anomaly behavior)")
        print("="*60)
        for field in self.fields:
            rates = self.calculate_rates(self.tagged_one[field])
            print(f"\n{field}:")
            sorted_rates = sorted(rates.items(), key=lambda x: x[1], reverse=True)[:5]
            for value, rate in sorted_rates:
                print(f"  {value}: {rate:.4f} ({rate*100:.2f}%)")
        
        # Find max rate attribute
        #max_field, max_value, max_rate = self.get_max_rate_attrb(self.message_input)
        max_field_z, max_value_z, max_rate_z = self.get_max_per_attrb("dur", self.tagged_z) 
        print(f"\n" + "="*60)
        #print(f"HIGHEST RATE ATTRIBUTE")
        print(f"Duration RATE ATTRIBUTE (Normal behavior)")
        print("="*60)
        print(f"Field: {max_field_z}")
        print(f"Value: {max_value_z}")
        print(f"Rate: {max_rate_z:.4f} ({max_rate_z*100:.2f}%)")
        
        # Find max rate attribute
        #max_field, max_value, max_rate = self.get_max_rate_attrb(self.message_input)
        max_field_o, max_value_o, max_rate_o = self.get_max_per_attrb("dur", self.tagged_one) 
        print(f"\n" + "="*60)
        #print(f"HIGHEST RATE ATTRIBUTE")
        print(f"Duration RATE ATTRIBUTE (Anomaly behavior)")
        print("="*60)
        print(f"Field: {max_field_o}")
        print(f"Value: {max_value_o}")
        print(f"Rate: {max_rate_o:.4f} ({max_rate_o*100:.2f}%)")
        
        # Conditional averages
        print(f"\n" + "="*60)
        print(f"CONDITIONAL AVERAGES (when {max_field_z} = {max_value_z})")
        print("="*60)
        cond_avgs = self.calculate_condt_avg(df, max_field_z, max_value_z)
        for field in sorted(cond_avgs.keys()):
            print(f"\n{field}:")
            print(f" Avg: {cond_avgs[field]:.4f} ({cond_avgs[field]*100:.2f}%)")
        
        # Conditional averages
        print(f"\n" + "="*60)
        print(f"CONDITIONAL AVERAGES (when {max_field_o} = {max_value_o})")
        print("="*60)
        cond_avgs = self.calculate_condt_avg(df, max_field_o, max_value_o)
        for field in sorted(cond_avgs.keys()):
            print(f"\n{field}:")
            print(f" Avg: {cond_avgs[field]:.4f} ({cond_avgs[field]*100:.2f}%)")
        
        # Tag-specific analysis
        for tag in [0, 1]:
            hist_dict = self.tagged_z if tag == 0 else self.tagged_one
            tag_max_field, tag_max_value, tag_max_rate = self.get_max_rate_attrb(hist_dict)
            
            print(f"\n" + "="*60)
            print(f"TAG {tag} - HIGHEST RATE ATTRIBUTE")
            print("="*60)
            print(f"Field: {tag_max_field}")
            print(f"Value: {tag_max_value}")
            print(f"Rate: {tag_max_rate:.4f} ({tag_max_rate*100:.2f}%)")
        
        return max_field_z, max_value_z, max_rate_z, cond_avgs
    
    def plot_analytics(self, df, max_field, max_value):
        """Generate analytical plots"""

        # Check if we have both tags
        tag_counts = df[self.tag_c].value_counts()
        has_tag0 = 0 in tag_counts and tag_counts[0] > 0
        has_tag1 = 1 in tag_counts and tag_counts[1] > 0

        if not has_tag0 and not has_tag1:
            print("Warning: No tagged data found. Skipping tag-based plots.")
            return

        fig = plt.figure(figsize=(20, 12))

        # Plot 1: Distribution of tags
        ax1 = plt.subplot(3, 3, 1)
        tag_counts = df[self.tag_c].value_counts()
        ax1.bar(['Tag 0', 'Tag 1'], [tag_counts.get(0, 0), tag_counts.get(1, 0)], 
                color=['blue', 'red'], alpha=0.7)
        ax1.set_title('Distribution of Tags', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Count')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Top rates for most common attribute (mixed)
        ax2 = plt.subplot(3, 3, 2)
        rates = self.calculate_rates(self.message_input[max_field])
        sorted_rates = sorted(rates.items(), key=lambda x: x[1], reverse=True)[:10]
        values, rate_vals = zip(*sorted_rates) if sorted_rates else ([], [])
        ax2.bar(range(len(values)), rate_vals, alpha=0.7, color='green')
        ax2.set_title(f'Top 10 Rates for {max_field} (Mixed)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Value Index')
        ax2.set_ylabel('Rate')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Comparison of max attribute across tags (only if both tags exist)
        ax3 = plt.subplot(3, 3, 3)
        if has_tag0 and has_tag1:
            rates_tag0 = self.calculate_rates(self.tagged_z[max_field])
            rates_tag1 = self.calculate_rates(self.tagged_one[max_field])

            # Get top values from mixed data
            top_values = [v for v, r in sorted_rates[:5]]
            tag0_rates = [rates_tag0.get(v, 0) for v in top_values]
            tag1_rates = [rates_tag1.get(v, 0) for v in top_values]

            x = np.arange(len(top_values))
            width = 0.35
            ax3.bar(x - width/2, tag0_rates, width, label='Tag 0', alpha=0.7, color='blue')
            ax3.bar(x + width/2, tag1_rates, width, label='Tag 1', alpha=0.7, color='red')
            ax3.set_title(f'{max_field} Rates: Tag 0 vs Tag 1', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Rate')
            ax3.set_xticks(x)
            ax3.set_xticklabels([f'V{i}' for i in range(len(top_values))], rotation=45)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, f'Only Tag {0 if has_tag0 else 1} present in data', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=12)

        # Plots 4-6: Distribution of numeric fields
        numeric_fields = [f for f in self.fields if pd.api.types.is_numeric_dtype(df[f])]
        for idx, field in enumerate(numeric_fields[:3], start=4):
            ax = plt.subplot(3, 3, idx)

            # Separate by tag
            data_tag0 = df[df[self.tag_c] == 0][field].dropna() if has_tag0 else pd.Series([])
            data_tag1 = df[df[self.tag_c] == 1][field].dropna() if has_tag1 else pd.Series([])

            if len(data_tag0) > 0:
                ax.hist(data_tag0, bins=30, alpha=0.6, color='blue', label='Tag 0')
            if len(data_tag1) > 0:
                ax.hist(data_tag1, bins=30, alpha=0.6, color='red', label='Tag 1')

            ax.set_title(f'{field} Distribution by Tag', fontsize=12, fontweight='bold')
            ax.set_xlabel(field)
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Plot 7: Heatmap of attribute rates by tag (only if we have data)
        ax7 = plt.subplot(3, 3, 7)
        try:
            # Get top 5 attributes with highest variance between tags
            variance_data = []
            for field in self.fields:
                rates_mixed = self.calculate_rates(self.message_input[field])
                rates_0 = self.calculate_rates(self.tagged_z[field]) if has_tag0 else {}
                rates_1 = self.calculate_rates(self.tagged_one[field]) if has_tag1 else {}
                
                max_mixed = max(rates_mixed.values()) if rates_mixed else 0
                max_0 = max(rates_0.values()) if rates_0 else 0
                max_1 = max(rates_1.values()) if rates_1 else 0
                
                variance = abs(max_0 - max_1) if has_tag0 and has_tag1 else max(max_0, max_1)
                variance_data.append((field, variance, max_mixed, max_0, max_1))
            
            variance_data.sort(key=lambda x: x[1], reverse=True)
            top_fields = [f for f, v, _, _, _ in variance_data[:5]]
            heatmap_data = np.array([[mixed, t0, t1] for _, _, mixed, t0, t1 in variance_data[:5]])
            
            if heatmap_data.size > 0:  # Check if array is not empty
                im = ax7.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
                ax7.set_xticks([0, 1, 2])
                ax7.set_xticklabels(['Mixed', 'Tag 0', 'Tag 1'])
                ax7.set_yticks(range(len(top_fields)))
                ax7.set_yticklabels(top_fields)
                ax7.set_title('Max Rates Heatmap', fontsize=12, fontweight='bold')
                plt.colorbar(im, ax=ax7)
            else:
                raise ValueError("Empty heatmap data")
                
        except (ValueError, TypeError):
            ax7.text(0.5, 0.5, 'Insufficient data for heatmap', 
                    ha='center', va='center', transform=ax7.transAxes, fontsize=12)

        # Get top 5 attributes with highest variance between tags
        variance_data = []
        for field in self.fields:
            rates_mixed = self.calculate_rates(self.message_input[field])
            rates_0 = self.calculate_rates(self.tagged_z[field]) if has_tag0 else {}
            rates_1 = self.calculate_rates(self.tagged_one[field]) if has_tag1 else {}

            # Calculate max rate for each
            max_mixed = max(rates_mixed.values()) if rates_mixed else 0
            max_0 = max(rates_0.values()) if rates_0 else 0
            max_1 = max(rates_1.values()) if rates_1 else 0

            # Use absolute difference as variance metric
            variance = abs(max_0 - max_1) if has_tag0 and has_tag1 else max(max_0, max_1)
            variance_data.append((field, variance, max_mixed, max_0, max_1))

        variance_data.sort(key=lambda x: x[1], reverse=True)
        top_fields = [f for f, v, _, _, _ in variance_data[:5]]
        heatmap_data = [[mixed, t0, t1] for _, _, mixed, t0, t1 in variance_data[:5]]

        if len(heatmap_data) > 0 and any(any(row) for row in heatmap_data):
            im = ax7.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
            ax7.set_xticks([0, 1, 2])
            ax7.set_xticklabels(['Mixed', 'Tag 0', 'Tag 1'])
            ax7.set_yticks(range(len(top_fields)))
            ax7.set_yticklabels(top_fields)
            ax7.set_title('Max Rates Heatmap', fontsize=12, fontweight='bold')
            plt.colorbar(im, ax=ax7)
        else:
            ax7.text(0.5, 0.5, 'Insufficient data for heatmap', 
                    ha='center', va='center', transform=ax7.transAxes)

        # Plot 8: Conditional vs Unconditional means
        ax8 = plt.subplot(3, 3, 8)
        cond_df = df[df[max_field] == max_value]

        comparison_fields = [f for f in numeric_fields if f != max_field][:5]

        if len(comparison_fields) > 0 and len(cond_df) > 0:
            unconditional_means = [df[f].mean() for f in comparison_fields]
            conditional_means = [cond_df[f].mean() for f in comparison_fields]

            x = np.arange(len(comparison_fields))
            width = 0.35
            ax8.bar(x - width/2, unconditional_means, width, label='Unconditional', alpha=0.7)
            ax8.bar(x + width/2, conditional_means, width, label=f'When {max_field}={max_value}', alpha=0.7)
            ax8.set_title('Conditional vs Unconditional Means', fontsize=12, fontweight='bold')
            ax8.set_ylabel('Mean Value')
            ax8.set_xticks(x)
            ax8.set_xticklabels(comparison_fields, rotation=45, ha='right')
            ax8.legend()
            ax8.grid(True, alpha=0.3)
        else:
            ax8.text(0.5, 0.5, 'Insufficient data for comparison', 
                    ha='center', va='center', transform=ax8.transAxes)

        # Plot 9: Sample count by tag over time (if time field exists)
        ax9 = plt.subplot(3, 3, 9)
        if 'stime' in df.columns and len(df) > 0:
            df_sorted = df.sort_values('stime')
            bins = min(20, len(df_sorted) // 10)  # Ensure we have enough data per bin

            if bins > 0:
                bin_size = len(df_sorted) // bins

                tag0_counts = []
                tag1_counts = []

                for i in range(bins):
                    start = i * bin_size
                    end = start + bin_size if i < bins - 1 else len(df_sorted)
                    chunk = df_sorted.iloc[start:end]
                    tag0_counts.append((chunk[self.tag_c] == 0).sum())
                    tag1_counts.append((chunk[self.tag_c] == 1).sum())

                x = range(bins)
                if has_tag0:
                    ax9.plot(x, tag0_counts, marker='o', label='Tag 0', alpha=0.7)
                if has_tag1:
                    ax9.plot(x, tag1_counts, marker='s', label='Tag 1', alpha=0.7)
                ax9.set_title('Tag Distribution Over Time Bins', fontsize=12, fontweight='bold')
                ax9.set_xlabel('Time Bin')
                ax9.set_ylabel('Count')
                ax9.legend()
                ax9.grid(True, alpha=0.3)
            else:
                ax9.text(0.5, 0.5, 'Insufficient data for time analysis', 
                        ha='center', va='center', transform=ax9.transAxes)
        else:
            ax9.text(0.5, 0.5, 'No time field available', 
                    ha='center', va='center', transform=ax9.transAxes)

        plt.tight_layout()
        plt.savefig('../analytical_plots.png', dpi=300, bbox_inches='tight')
        print("\nPlots saved to 'analytical_plots.png'")
        plt.close()

        
def main():
    fields=["proto_number", "saddr", "daddr", "dport", "state_number", "dur", "spkts", "sbytes"]
    # Lire le CSV
    df = pd.read_csv('../databaseDoS.csv')
    print(f"Données totales: {len(df)} lignes")
    
    tag_column = df.columns[-1]  # 15th column (0-indexed)
    print(f"Tag column: {tag_column}")
    
    quart = len(df) // 4
    
    df_train = df.iloc[:quart]
    
    # Create analyzer
    analyzer = History_analyzer(fields=fields, tag_c=str(tag_column))
    
    # Update histograms with training data
    analyzer.update_histograms(df)
    
    # Generate reports
    max_field, max_value, max_rate, cond_avgs = analyzer.generate_reports(df_train)
    
    # Generate plots
    analyzer.plot_analytics(df_train, max_field, max_value)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    
    return analyzer, max_field, max_value, cond_avgs

    
if __name__ == "__main__":
    analyzer, max_field, max_value, cond_avgs = main()