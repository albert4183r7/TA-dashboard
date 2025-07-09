import pandas as pd
import matplotlib.pyplot as plt

# Konfigurasi file
file_path = 'Training-conventionalsac-45cars-500seconds-omnidirectional.xlsx'  # Ganti dengan path file Anda
sheet_name = 'Detailed_Results'  # Ganti dengan nama sheet yang sesuai

try:
    # Baca data dari Excel
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    
    # Verifikasi kolom yang diperlukan
    required_columns = ['Timestamp', 'CBR', 'SINR', 'Latency', 'PDR']
    missing_cols = [col for col in required_columns if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Kolom berikut tidak ditemukan: {missing_cols}")

    # Hitung rata-rata per timestamp
    avg_data = df.groupby('Timestamp').agg({
        'CBR': 'mean',
        'SINR': 'mean',
        'Latency': 'mean',
        'PDR': 'mean'
    }).reset_index()

    # Urutkan berdasarkan timestamp
    avg_data.sort_values('Timestamp', inplace=True)

    # Buat 4 plot terpisah
    plt.figure(figsize=(15, 12))

    # 1. Plot CBR
    plt.subplot(2, 2, 1)
    plt.plot(avg_data['Timestamp'], avg_data['CBR'], 'b-', marker='o')
    plt.title('Rata-rata CBR per Timestamp')
    plt.ylabel('CBR')
    plt.grid(True)
    plt.xticks(rotation=45)

    # 2. Plot SINR
    plt.subplot(2, 2, 2)
    plt.plot(avg_data['Timestamp'], avg_data['SINR'], 'r-', marker='s')
    plt.title('Rata-rata SINR per Timestamp')
    plt.ylabel('SINR (dB)')
    plt.grid(True)
    plt.xticks(rotation=45)

    # 3. Plot Latency
    plt.subplot(2, 2, 3)
    plt.plot(avg_data['Timestamp'], avg_data['Latency'], 'g-', marker='^')
    plt.title('Rata-rata Latency per Timestamp')
    plt.ylabel('Latency (ms)')
    plt.xlabel('Timestamp')
    plt.grid(True)
    plt.xticks(rotation=45)

    # 4. Plot PDR
    plt.subplot(2, 2, 4)
    plt.plot(avg_data['Timestamp'], avg_data['PDR'], 'm-', marker='d')
    plt.title('Rata-rata PDR per Timestamp')
    plt.ylabel('PDR (%)')
    plt.xlabel('Timestamp')
    plt.grid(True)
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig('all_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Buat juga plot individual untuk masing-masing metrik
    for metric in ['CBR', 'SINR', 'Latency', 'PDR']:
        plt.figure(figsize=(10, 5))
        plt.plot(avg_data['Timestamp'], avg_data[metric], '-', marker='o')
        plt.title(f'Rata-rata {metric} per Timestamp')
        plt.ylabel(f'{metric}' + (' (dB)' if metric == 'SINR' else 
                                ' (ms)' if metric == 'Latency' else
                                ' (%)' if metric == 'PDR' else ''))
        plt.xlabel('Timestamp')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'avg_{metric.lower()}.png', dpi=300, bbox_inches='tight')
        plt.show()

    # Tampilkan data rata-rata
    print("\nData Rata-rata per Timestamp:")
    print(avg_data)

except FileNotFoundError:
    print(f"File tidak ditemukan: {file_path}")
except Exception as e:
    print(f"Terjadi kesalahan: {str(e)}")