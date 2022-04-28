packages <- c("dplyr", "stringi")
installed_packages <- packages %in% rownames(installed.packages())
if (any(installed_packages == FALSE)) {
  install.packages(packages[!installed_packages])
}
invisible(lapply(packages, library, character.only = TRUE))

colorectal <- read.csv('colorectal.csv', header = TRUE, sep = ";", na = c(""," "))
esophageal <- read.csv('esophageal.csv', header = TRUE, sep = ";", na = c(""," "))
prostate <- read.csv('prostate.csv', header = TRUE, sep = ";", na = c(""," "))

# -----------------------------
# ADJUSTING & DELETING THE COLUMNS

# -----------------------------
# Cleaning the esophageal file

# Re-naming necessary columns
names(colorectal)[names(colorectal) == "Colonoscopy"] = "Examination"
namescolo <- names(colorectal)
namesesop <- names(esophageal)

wrongesopnames <- vector(mode = "list", length = 0)
for (i in 1:ncol(esophageal)) {
  if (!(namesesop[i] %in% namescolo)) {
    wrongesopnames <- append(wrongesopnames, namesesop[i])
  }
}

wrongesopnames <- wrongesopnames[c(1,2,3,4,7,13,14,15)]
correctesopnames <- c("Date_Dx", "Edu_status", "MStatus", "Family_history", "weight_loss", "Examination", "Imaging", "Dist_metastasis")

for (i in 1:length(correctesopnames)) names(esophageal)[names(esophageal) == wrongesopnames[i]] = correctesopnames[i]

# Drop columns of esophageal that are not present in colorectal
dropnames <- vector(mode = "list", length = 0)
for (i in 1:ncol(esophageal)) {
  if (!(names(esophageal)[i] %in% names(colorectal))) {
    dropnames <- append(dropnames, names(esophageal)[i])
  }
}
esophageal <- esophageal[, !(names(esophageal) %in% dropnames)]

# Drop columns of colorectal that are not present in esophageal
dropnames1 <- vector(mode = "list", length = 0)
for (i in 1:ncol(colorectal)) {
  if (!(names(colorectal)[i] %in% names(esophageal))) {
    dropnames1 <- append(dropnames1, names(colorectal)[i])
  }
}
colorectal <- colorectal[, !(names(colorectal) %in% dropnames1)]


# -----------------------------
# Cleaning the prostate file

# Typo during data collection- change names of 'region'
prostate$region <- replace(prostate$region, prostate$region == 'OROMIYA', 'OROMIA')

# Re-naming the columns
namescolo <- names(colorectal)
namespros <- names(prostate)

wrongprosnames <- vector(mode = "list", length = 0)
for (i in 1:ncol(prostate)) {
  if (!(namespros[i] %in% namescolo)) {
    wrongprosnames <- append(wrongprosnames, namespros[i])
  }
}

wrongprosnames <- wrongprosnames[c(1,2,3,4)]
correctprosnames <- c("Region", "Edu_status", "MStatus", "Age")

for (i in 1:length(correctprosnames)) names(prostate)[names(prostate) == wrongprosnames[i]] = correctprosnames[i]

# Drop columns of prostate that are not present in colorectal
dropnames2 <- vector(mode = "list", length = 0)
for (i in 1:ncol(prostate)) {
  if (!(names(prostate)[i] %in% names(colorectal))) {
    dropnames2 <- append(dropnames2, names(prostate)[i])
  }
}
prostate <- prostate[, !(names(prostate) %in% dropnames2)]

# Drop columns of colorectal & esophageal that are not present in prostate
dropnames3 <- vector(mode = "list", length = 0)
for (i in 1:ncol(colorectal)) {
  if (!(names(colorectal)[i] %in% names(prostate))) {
    dropnames3 <- append(dropnames3, names(colorectal)[i])
  }
}
colorectal <- colorectal[, !(names(colorectal) %in% dropnames3)]
esophageal <- esophageal[, !(names(esophageal) %in% dropnames3)]


# Re-ordering the columns of esophageal & prostate
col_order <- names(colorectal)
esophageal <- esophageal[, col_order]
prostate <- prostate[, col_order]

# -----------------------------
# ADJUSTING DUMMY VARIABLE DISCREPANCIES (ex. 0-1 vs 1-2)

# esophageal
esophageal$Alcohol <- esophageal$Alcohol - 1
esophageal$Tobacco <- esophageal$Tobacco - 1
esophageal$Khat <- esophageal$Khat - 1
esophageal$Status_patient <- esophageal$Status_patient - 1
esophageal$Region <- stri_trans_totitle(esophageal$Region)

for (i in 1:nrow(esophageal)) {
  if (esophageal[i,15] == 1) {
    esophageal[i,15] = 0
  } else {esophageal[i,15] = 1}
}

# prostate
prostate$Family_history <- prostate$Family_history - 2
prostate$Dist_metastasis <- prostate$Dist_metastasis - 1
prostate$Chemotherapy <- prostate$Chemotherapy - 1
prostate$Radiotherapy <- prostate$Radiotherapy - 1
prostate$Status_patient <- prostate$Status_patient - 1
prostate$Region <- stri_trans_totitle(prostate$Region)

for (i in 1:nrow(prostate)) {
  if (prostate[i,8] == 1) {
    prostate[i,8] = 0
  } else {prostate[i,8] = 1}
}

# colorectal
colorectal$Region <- stri_trans_totitle(colorectal$Region)

for (i in 1:nrow(colorectal)) {
  if (is.na(colorectal[i, 17])){
    colorectal[i, 17] = 0
  }
  else {colorectal[i, 17] = 1}
}

# Add columns to each dataframe indicating cancer type

esophageal$Cancer_type <- "esophageal"
esophageal <- esophageal %>% relocate(Cancer_type, .before = MStatus)

prostate$Cancer_type <- "prostate"
prostate <- prostate %>% relocate(Cancer_type, .before = MStatus)

colorectal$Cancer_type <- "colorectal"
colorectal <- colorectal %>% relocate(Cancer_type, .before = MStatus)


# -----------------------------
# MERGING ALL DATA FRAMES INTO ONE & SAVING

fulldata <- rbind(colorectal, esophageal, prostate)

write.csv(fulldata, "cancerdatamerged.csv", row.names = FALSE)

