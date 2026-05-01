

library(dplyr)
library(ggplot2)
library(scales)
library(viridis)
library(gridExtra)
library(grid)

# ── Load Data ─────────────────────────────────────────────────────────────────
if (file.exists("df_cleaned.rds")) {
  df <- readRDS("df_cleaned.rds")
} else {
  df <- read.csv("C:/Users/HP/Downloads/AccidentsBig.csv", stringsAsFactors = FALSE)
}

# ── Fix Date + Weekend ─────────────────────────────
df$date <- as.Date(df$date)

df$weekend_lbl <- ifelse(weekdays(df$date) %in% c("Saturday","Sunday"),
                         "Weekend","Weekday")

# ── Accident Seriousness ───────────────────────────
df$accident_seriousness <- ifelse(df$accident_severity == 3,
                                  "Not Serious","Serious")

# ── Decode columns ─────────────────────────────────

# Police attended
df$police_attended_lbl <- recode(as.character(df$police_attended),
                                 "1"="Yes","2"="No","3"="Self-Reported")

# Urban / Rural
df$urban_rural_lbl <- recode(as.character(df$urban_or_rural_area),
                             "1"="Urban","2"="Rural","3"="Unallocated")

# Time of day (from time column)
df$time_of_day_lbl <- ifelse(as.numeric(substr(df$time,1,2)) < 10, "Morning",
                             ifelse(as.numeric(substr(df$time,1,2)) < 16, "Day",
                                    ifelse(as.numeric(substr(df$time,1,2)) < 20, "Evening",
                                           "Night")))

# Light conditions
df$light_lbl <- recode(as.character(df$light_conditions),
                       "1"="Daylight","4"="Dark: lit","5"="Dark: unlit",
                       "6"="Dark: no light","7"="Dark: unknown")

# Weather
df$weather_lbl <- recode(as.character(df$weather_conditions),
                         "1"="Fine","2"="Raining","3"="Snowing",
                         "4"="Fine+wind","5"="Rain+wind","6"="Snow+wind",
                         "7"="Fog","8"="Other","9"="Unknown")

# Road surface
df$surface_lbl <- recode(as.character(df$road_surface_conditions),
                         "1"="Dry","2"="Wet","3"="Snow","4"="Frost/Ice",
                         "5"="Flood","6"="Oil/Diesel","7"="Mud")

# Junction control
df$jc_lbl <- recode(as.character(df$junction_control),
                    "0"="Not at junction","1"="Authorised person",
                    "2"="Auto signal","3"="Stop sign","4"="Give way")

# Junction detail
df$jd_lbl <- recode(as.character(df$junction_detail),
                    "0"="Not at junction","1"="Roundabout","2"="Mini-roundabout",
                    "3"="T/Staggered junction","5"="Slip road","6"="Crossroads",
                    "7"=">4 arms","8"="Private drive","9"="Other")

# Direct columns
df$road_type <- as.character(df$road_type)
df$speed_limit <- as.character(df$speed_limit)
df$carriageway_hazards <- as.character(df$carriageway_hazards)
df$special_conditions <- as.character(df$special_conditions)

# ── Split subsets ──────────────────────────────────
not_serious <- df %>% filter(accident_seriousness == "Not Serious")
serious <- df %>% filter(accident_seriousness == "Serious")

ns_total <- nrow(not_serious)
se_total <- nrow(serious)

# ── Helper function (IMPORTANT FIX: SHOW + SAVE) ──────────────────────────────
seriousness_plot <- function(df_ns, df_se, col, title,
                             xlab_ns, xlab_se,
                             ns_total, se_total,
                             fname=NULL) {
  
  make_panel <- function(data, total, xlab) {
    cnt <- data %>% count(.data[[col]])
    cnt$pct <- cnt$n / total * 100
    
    ggplot(cnt, aes(x=.data[[col]], y=n, fill=.data[[col]],
                    label=sprintf("%.1f%%", pct))) +
      geom_col(show.legend=FALSE) +
      geom_text(vjust=-0.4, size=3.5) +
      scale_fill_viridis_d(option="plasma") +
      scale_y_continuous(labels=comma,
                         expand=expansion(mult=c(0,0.13))) +
      labs(x=xlab, y="Count") +
      theme_minimal() +
      theme(axis.text.x=element_text(angle=35, hjust=1))
  }
  
  p1 <- make_panel(df_ns, ns_total, xlab_ns)
  p2 <- make_panel(df_se, se_total, xlab_se)
  
  combined <- gridExtra::arrangeGrob(
    p1, p2, nrow = 2,
    top = grid::textGrob(title,
                         gp = grid::gpar(fontsize = 16, fontface = "bold"))
  )
  
  grid::grid.newpage()
  grid::grid.draw(combined)
  # 🔥 SAVE PLOT
  if (!is.null(fname)) {
    ggsave(fname, combined, width = 12, height = 8)
    cat("Saved:", fname, "\n")
  }
}

# ── Generate Plots ────────────────────────────────────────────────────────────
seriousness_plot(not_serious, serious, "police_attended_lbl",
                 "Police Attendance","Not Serious","Serious",
                 ns_total,se_total,"police.png")

seriousness_plot(not_serious, serious, "urban_rural_lbl",
                 "Urban vs Rural","Not Serious","Serious",
                 ns_total,se_total,"urban.png")

seriousness_plot(not_serious, serious, "speed_limit",
                 "Speed Limit","Not Serious","Serious",
                 ns_total,se_total,"speed.png")

seriousness_plot(not_serious, serious, "weekend_lbl",
                 "Weekend vs Weekday","Not Serious","Serious",
                 ns_total,se_total,"weekend.png")

seriousness_plot(not_serious, serious, "time_of_day_lbl",
                 "Time of Day","Not Serious","Serious",
                 ns_total,se_total,"time.png")

seriousness_plot(not_serious, serious, "light_lbl",
                 "Light Conditions","Not Serious","Serious",
                 ns_total,se_total,"light.png")

seriousness_plot(not_serious, serious, "weather_lbl",
                 "Weather","Not Serious","Serious",
                 ns_total,se_total,"weather.png")

seriousness_plot(not_serious, serious, "surface_lbl",
                 "Road Surface","Not Serious","Serious",
                 ns_total,se_total,"surface.png")

seriousness_plot(not_serious, serious, "jc_lbl",
                 "Junction Control","Not Serious","Serious",
                 ns_total,se_total,"jc.png")

seriousness_plot(not_serious, serious, "jd_lbl",
                 "Junction Detail","Not Serious","Serious",
                 ns_total,se_total,"jd.png")

seriousness_plot(not_serious, serious, "carriageway_hazards",
                 "Carriageway Hazards","Not Serious","Serious",
                 ns_total,se_total,"hazards.png")

seriousness_plot(not_serious, serious, "special_conditions",
                 "Special Conditions","Not Serious","Serious",
                 ns_total,se_total,"special.png")

cat("\n✅ ALL PLOTS GENERATED + DISPLAYED SUCCESSFULLY\n")


cat("\n================================================\n")
cat("  SERIOUSNESS VISUALIZATIONS COMPLETE\n")
cat("  Seriousness comparison plots:\n")
cat("    police_attended.png\n")
cat("    urban_or_rural_area.png\n")
cat("    number_of_vehicles.png\n")
cat("    number_of_casualties.png\n")
cat("    speed_limit.png\n")
cat("    weekend_vs_weekday.png\n")
cat("    time_of_day.png\n")
cat("    season.png\n")
cat("    road_type.png\n")
cat("    junction_control.png\n")
cat("    junction_detail.png\n")
cat("    light_conditions.png\n")
cat("    weather_conditions.png\n")
cat("    road_surface_conditions.png\n")
cat("    carriageway_hazards.png\n")
cat("    special_conditions.png\n")
cat("  Cross-feature plots:\n")
cat("    junction_control_by_junction_detail.png\n")
cat("    junction_control_by_road_type.png\n")
cat("    speed_limit_by_road_type.png\n")
cat("    weather_by_road_surface.png\n")
cat("    light_by_weather.png\n")
cat("    junction_control_by_weekend.png\n")
cat("    tod_by_urban_rural.png\n")
cat("================================================\n") 