packages <- c("missForest")
installed_packages <- packages %in% rownames(installed.packages())
if (any(installed_packages == FALSE)) {
  install.packages(packages[!installed_packages])
}
invisible(lapply(packages, library, character.only = TRUE))

cancerdatamerged <- read.csv("cancerdatamerged.csv", header = TRUE)
cancerdatanumeric <- cancerdatamerged[-c(1, 3, 4, 17, 19)]

# Use Random Forest to estimate values for NA
cancerimpute <- missForest(cancerdatanumeric)
imputedcancer <- round(cancerimpute$ximp)

# Plug values back into dataset

for (var in names(imputedcancer)){
  for (i in 1:nrow(cancerdatamerged)) {
    if (is.na(cancerdatamerged[[var]][i])) {cancerdatamerged[[var]][i] = imputedcancer[[var]][i]}
  }
}

write.csv(cancerdatamerged, "finalcancerdata.csv", row.names = FALSE)
