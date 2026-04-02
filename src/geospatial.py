import folium
from folium.plugins import HeatMap
import os


def create_pollution_map(df, output_dir="outputs/"):
    """
    Build an interactive Folium heatmap showing WQI across the Niger Delta.
    Saves to outputs/pollution_map.html
    """
    os.makedirs(output_dir, exist_ok=True)

    # Check if WQI exists, if not use Heavy_Metal_Index as fallback
    if "WQI" not in df.columns:
        print("Warning: WQI not found, using Heavy_Metal_Index for visualization")
        if "Heavy_Metal_Index" not in df.columns:
            raise ValueError("Neither WQI nor Heavy_Metal_Index found in dataframe")
        value_col = "Heavy_Metal_Index"
        # For pollution index, higher = worse, so invert for visualization
        df = df.copy()
        df["WQI_viz"] = 100 - (df["Heavy_Metal_Index"] * 100).clip(0, 100)
        wqi_col = "WQI_viz"
    else:
        wqi_col = "WQI"
        value_col = "WQI"

    # Drop rows with missing coordinates or values
    df_clean = df.dropna(subset=["Latitude", "Longitude", wqi_col]).copy()

    if len(df_clean) == 0:
        raise ValueError("No valid data points with coordinates")

    # Center map on data mean (or Niger Delta default)
    if df_clean["Latitude"].notna().any() and df_clean["Longitude"].notna().any():
        center_lat = df_clean["Latitude"].mean()
        center_lon = df_clean["Longitude"].mean()
    else:
        center_lat, center_lon = 5.3, 6.5  # Niger Delta default

    m = folium.Map(location=[center_lat, center_lon], zoom_start=7, tiles="CartoDB positron")

    # --- Heatmap Layer ---
    heat_data = [
        [row["Latitude"], row["Longitude"], row[wqi_col]]
        for _, row in df_clean.iterrows()
    ]

    if len(heat_data) > 0:
        HeatMap(heat_data, radius=15, blur=10, max_zoom=1).add_to(m)

    # --- Individual Markers ---
    for _, row in df_clean.iterrows():
        try:
            wqi_val = row[wqi_col]
            
            # WHO 5-tier scale: higher WQI = better (greener)
            if value_col == "WQI":
                if wqi_val >= 95:
                    color = "green"
                    status = "Excellent"
                elif wqi_val >= 80:
                    color = "blue"
                    status = "Good"
                elif wqi_val >= 65:
                    color = "beige"
                    status = "Fair"
                elif wqi_val >= 45:
                    color = "orange"
                    status = "Poor"
                else:
                    color = "red"
                    status = "Hazardous"
            else:
                # For pollution-based visualization
                color = "red" if wqi_val < 25 else "orange" if wqi_val < 50 else "green"
                status = "High Pollution" if wqi_val < 25 else "Moderate" if wqi_val < 50 else "Low"

            folium.CircleMarker(
                location=[row["Latitude"], row["Longitude"]],
                radius=6,
                color="blue",
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=folium.Popup(
                    f"<b>WQI:</b> {wqi_val:.2f} ({status})<br>"
                    f"<b>River:</b> {row.get('River_Name', 'N/A')}<br>"
                    f"<b>State:</b> {row.get('State', 'N/A')}<br>"
                    f"<b>Station:</b> {row.get('Station_Name', 'N/A')}",
                    max_width=250
                )
            ).add_to(m)
        except Exception as e:
            continue

    output_path = os.path.join(output_dir, "pollution_map.html")
    m.save(output_path)
    print(f"Pollution map saved to {output_path}")

    return m