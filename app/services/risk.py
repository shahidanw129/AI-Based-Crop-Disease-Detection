def assess_crop_risk(crop_name, humidity_percent, rainfall_mm):
    crop = crop_name.casefold()
    humidity = int(humidity_percent)
    rainfall = float(rainfall_mm)
    if crop in {"tomato", "potato"}:
        watch = humidity >= 80 and rainfall >= 1
        elevated = humidity >= 90 and rainfall >= 3
        concern = "conditions associated with foliar blight pressure"
    elif crop in {"corn", "maize"}:
        watch = humidity >= 85 and rainfall >= 2
        elevated = humidity >= 92 and rainfall >= 5
        concern = "prolonged leaf-wetness conditions"
    else:
        watch = humidity >= 85 and rainfall >= 3
        elevated = humidity >= 93 and rainfall >= 8
        concern = "humid, wet field conditions"

    if elevated:
        level = "elevated"
        message = (
            f"Humidity ({humidity}%) and recent precipitation ({rainfall:g} mm) may favour {concern}. "
            "Scout several field locations, avoid unnecessary leaf wetness, and consult local crop guidance."
        )
    elif watch:
        level = "watch"
        message = (
            f"Humidity ({humidity}%) or recent precipitation ({rainfall:g} mm) merits closer scouting for {crop_name}. "
            "Check leaves and local advisories before changing field practices."
        )
    else:
        level = "low"
        message = "Current humidity and rainfall do not cross this project's simple weather watch thresholds. Continue routine crop scouting."
    return {"crop_name": crop_name, "level": level, "title": f"{crop_name} weather watch: {level}", "message": message}