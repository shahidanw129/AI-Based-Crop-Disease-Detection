CREATE DATABASE IF NOT EXISTS SmartFarmingDB
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE SmartFarmingDB;

-- The Flask application creates and manages tables from app/models.py.
-- This file documents the tables for MySQL Workbench and ERD preparation.
CREATE TABLE IF NOT EXISTS user (
  id INT AUTO_INCREMENT PRIMARY KEY,
  name VARCHAR(100) NOT NULL,
  email VARCHAR(255) NOT NULL UNIQUE,
  password_hash VARCHAR(255) NOT NULL,
  role VARCHAR(20) NOT NULL DEFAULT 'user',
  city VARCHAR(100) NOT NULL DEFAULT '',
  is_active BOOLEAN NOT NULL DEFAULT TRUE,
  created_at DATETIME NOT NULL
);

CREATE TABLE IF NOT EXISTS disease (
  id INT AUTO_INCREMENT PRIMARY KEY,
  crop_name VARCHAR(80) NOT NULL,
  name VARCHAR(140) NOT NULL UNIQUE,
  symptoms TEXT NOT NULL,
  prevention TEXT NOT NULL,
  INDEX ix_disease_crop_name (crop_name)
);

CREATE TABLE IF NOT EXISTS detection (
  id INT AUTO_INCREMENT PRIMARY KEY,
  user_id INT NOT NULL,
  disease_id INT NULL,
  image_path VARCHAR(255) NOT NULL,
  predicted_label VARCHAR(140) NOT NULL,
  confidence FLOAT NOT NULL,
  created_at DATETIME NOT NULL,
  INDEX ix_detection_user_id (user_id),
  INDEX ix_detection_created_at (created_at),
  CONSTRAINT fk_detection_user FOREIGN KEY (user_id) REFERENCES user(id) ON DELETE CASCADE,
  CONSTRAINT fk_detection_disease FOREIGN KEY (disease_id) REFERENCES disease(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS farming_tip (
  id INT AUTO_INCREMENT PRIMARY KEY,
  crop_name VARCHAR(80) NOT NULL DEFAULT 'All crops',
  title VARCHAR(140) NOT NULL,
  body TEXT NOT NULL,
  created_at DATETIME NOT NULL
);

CREATE TABLE IF NOT EXISTS weather_log (
  id INT AUTO_INCREMENT PRIMARY KEY,
  user_id INT NOT NULL,
  city VARCHAR(100) NOT NULL,
  temperature_c FLOAT NOT NULL,
  humidity_percent INT NOT NULL,
  rainfall_mm FLOAT NOT NULL DEFAULT 0,
  created_at DATETIME NOT NULL,
  INDEX ix_weather_log_user_id (user_id),
  CONSTRAINT fk_weather_log_user FOREIGN KEY (user_id) REFERENCES user(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS farm (
  id INT AUTO_INCREMENT PRIMARY KEY,
  user_id INT NOT NULL,
  name VARCHAR(120) NOT NULL,
  area_hectares FLOAT NOT NULL,
  location VARCHAR(160) NOT NULL DEFAULT '',
  created_at DATETIME NOT NULL,
  INDEX ix_farm_user_id (user_id),
  CONSTRAINT fk_farm_user FOREIGN KEY (user_id) REFERENCES user(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS plot (
  id INT AUTO_INCREMENT PRIMARY KEY,
  farm_id INT NOT NULL,
  name VARCHAR(120) NOT NULL,
  crop_name VARCHAR(80) NOT NULL,
  planting_date DATE NOT NULL,
  growth_stage VARCHAR(80) NOT NULL DEFAULT 'Seedling',
  INDEX ix_plot_farm_id (farm_id),
  CONSTRAINT fk_plot_farm FOREIGN KEY (farm_id) REFERENCES farm(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS plot_detection (
  id INT AUTO_INCREMENT PRIMARY KEY,
  plot_id INT NOT NULL,
  detection_id INT NOT NULL UNIQUE,
  created_at DATETIME NOT NULL,
  INDEX ix_plot_detection_plot_id (plot_id),
  CONSTRAINT fk_plot_detection_plot FOREIGN KEY (plot_id) REFERENCES plot(id) ON DELETE CASCADE,
  CONSTRAINT fk_plot_detection_detection FOREIGN KEY (detection_id) REFERENCES detection(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS risk_alert (
  id INT AUTO_INCREMENT PRIMARY KEY,
  user_id INT NOT NULL,
  weather_log_id INT NULL,
  city VARCHAR(100) NOT NULL,
  crop_name VARCHAR(80) NOT NULL,
  risk_level VARCHAR(20) NOT NULL,
  title VARCHAR(140) NOT NULL,
  message TEXT NOT NULL,
  created_at DATETIME NOT NULL,
  INDEX ix_risk_alert_user_id (user_id),
  INDEX ix_risk_alert_created_at (created_at),
  CONSTRAINT fk_risk_alert_user FOREIGN KEY (user_id) REFERENCES user(id) ON DELETE CASCADE,
  CONSTRAINT fk_risk_alert_weather FOREIGN KEY (weather_log_id) REFERENCES weather_log(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS expert_profile (
  id INT AUTO_INCREMENT PRIMARY KEY,
  name VARCHAR(120) NOT NULL,
  organization VARCHAR(160) NOT NULL DEFAULT '',
  expertise VARCHAR(200) NOT NULL,
  region VARCHAR(120) NOT NULL DEFAULT '',
  contact_email VARCHAR(255) NOT NULL DEFAULT '',
  is_verified BOOLEAN NOT NULL DEFAULT FALSE,
  is_active BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS consultation (
  id INT AUTO_INCREMENT PRIMARY KEY,
  user_id INT NOT NULL,
  expert_id INT NOT NULL,
  detection_id INT NULL,
  report_consent BOOLEAN NOT NULL DEFAULT FALSE,
  question TEXT NOT NULL,
  response TEXT NOT NULL,
  status VARCHAR(30) NOT NULL DEFAULT 'Requested',
  created_at DATETIME NOT NULL,
  updated_at DATETIME NOT NULL,
  INDEX ix_consultation_user_id (user_id),
  INDEX ix_consultation_created_at (created_at),
  CONSTRAINT fk_consultation_user FOREIGN KEY (user_id) REFERENCES user(id) ON DELETE CASCADE,
  CONSTRAINT fk_consultation_expert FOREIGN KEY (expert_id) REFERENCES expert_profile(id),
  CONSTRAINT fk_consultation_detection FOREIGN KEY (detection_id) REFERENCES detection(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS reminder (
  id INT AUTO_INCREMENT PRIMARY KEY,
  user_id INT NOT NULL,
  farm_id INT NULL,
  title VARCHAR(140) NOT NULL,
  body TEXT NOT NULL,
  due_at DATETIME NOT NULL,
  completed_at DATETIME NULL,
  notified_at DATETIME NULL,
  created_at DATETIME NOT NULL,
  INDEX ix_reminder_user_id (user_id),
  INDEX ix_reminder_due_at (due_at),
  CONSTRAINT fk_reminder_user FOREIGN KEY (user_id) REFERENCES user(id) ON DELETE CASCADE,
  CONSTRAINT fk_reminder_farm FOREIGN KEY (farm_id) REFERENCES farm(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS notification (
  id INT AUTO_INCREMENT PRIMARY KEY,
  user_id INT NOT NULL,
  kind VARCHAR(40) NOT NULL,
  title VARCHAR(140) NOT NULL,
  body TEXT NOT NULL,
  target_url VARCHAR(255) NOT NULL DEFAULT '',
  read_at DATETIME NULL,
  created_at DATETIME NOT NULL,
  INDEX ix_notification_user_id (user_id),
  INDEX ix_notification_created_at (created_at),
  CONSTRAINT fk_notification_user FOREIGN KEY (user_id) REFERENCES user(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS model_version (
  id INT AUTO_INCREMENT PRIMARY KEY,
  name VARCHAR(100) NOT NULL,
  version VARCHAR(80) NOT NULL,
  architecture VARCHAR(80) NOT NULL,
  model_path VARCHAR(500) NOT NULL UNIQUE,
  labels_path VARCHAR(500) NOT NULL,
  metrics_json TEXT NOT NULL,
  is_active BOOLEAN NOT NULL DEFAULT FALSE,
  created_at DATETIME NOT NULL
);

CREATE TABLE IF NOT EXISTS prediction_performance (
  id INT AUTO_INCREMENT PRIMARY KEY,
  detection_id INT NOT NULL UNIQUE,
  model_version_id INT NULL,
  inference_ms FLOAT NOT NULL,
  confidence FLOAT NOT NULL,
  predicted_label VARCHAR(140) NOT NULL,
  actual_label VARCHAR(140) NULL,
  created_at DATETIME NOT NULL,
  INDEX ix_prediction_performance_model_version_id (model_version_id),
  INDEX ix_prediction_performance_created_at (created_at),
  CONSTRAINT fk_prediction_performance_detection FOREIGN KEY (detection_id) REFERENCES detection(id) ON DELETE CASCADE,
  CONSTRAINT fk_prediction_performance_model FOREIGN KEY (model_version_id) REFERENCES model_version(id)
);