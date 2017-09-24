# Ler dados
imoveis <- read.csv(file="Remax data_revised_susana.csv", sep=";", dec=",", strip.white = TRUE)
# Fazer Sys.setlocale(category = "LC_ALL", locale = "pt_PT.UTF-8") para exibir caracteres correctamente
table(imoveis$erate)
imo<-subset(imoveis, erate!="NC" & erate!="")
# ou
#imo<-imoveis[imoveis["erate"]!="",]
#imo<-imo[imo["erate"]!="NC",]
table(imo$erate)
imo$preco <- gsub("[^0-9]", "", imo$preco)
imo$preco <- as.numeric(imo$preco)
anova(lm(imo$preco~imo$erate))
## H0: 