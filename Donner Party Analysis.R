#import libraries
library(dplyr)
library(tidyr)
library(magrittr)
library(knitr)
library(ggsci)
library(vcd)
library(vcdExtra)

#install packages
install.packages("vcdExtra")
install.packages("rmarkdown")

# Load the Donner Party data set
data("Donner",package="vcdExtra")
head(Donner,n=90L)

#ratio between male vs female can be calculated through this info

library(ggplot2)
library(ggpubr)

library(ggplot2)
summary(Donner)

#total number of survivors in each family
df = data.frame("family" = c("Breen", "Donner", "Eddy", "Fosdwold", "MurFosPik", "Graves", "Keseberg", "Reed","McCutchen"))

#number of children below the age of 18
num_children <-sum(Donner$age<18)
num_children <-sum(Donner$age>=18, na.rm = TRUE)
print(num_children)

#check for accuracy calculating children 18 or older
num_children <-sum(Donner$age>=18)
print(num_children)

#check for accuracy: sum should equal to total population
#sum equals the total of 90

#most common first name for females
female_names <- Donner[Donner$Sex=="female",]

#P1
#Own Question 1
#What percentage of children survived?
children <- Donner %>%
  filter(age < 18) %>%
  select(survived)

children_lived <- colSums(children == 1)
tot_children <- nrow(children)
percent_children_lived <- children_lived / tot_children

print(paste("Percent of children that survived: ", percent_children_lived *100))
#answer: 65.8536585365854

#Own question 2
#What is the survival rate for males?
######### survival rate for males
males <- Donner %>%
  filter(sex == "Male") %>%
  select(survived)

males_lived <- colSums(males == 1)
tot_m <- nrow(males)
percent_m_survived <- males_lived / tot_m
print(paste("Percent of men that survived: ", percent_m_survived *100))
#answer: 41.8181818181818

#Own Question 3
#What is the survival rate for females?
females <- Donner %>%
  filter(sex == "Female") %>%
  select(survived)


females_lived <- colSums(females == 1)
tot_f = nrow(females)
percent_f_survived <- females_lived / tot_f
print(paste("Percent of women that survived: ", percent_f_survived*100))   
#answer:71.4285714285714

#Own question 4
#Plot the relationship between the age of 10 Donner members and their survival
random10 <- Donner[sample(nrow(Donner), 10), ]
random10 <- random10 %>%
  mutate(names = row.names(random10)) %>%
  mutate(survived = factor(survived))
random10

#plot
g <- ggbarplot(random10, x = "names", y = "age", 
               color = "black",
               fill = "survived",
               palette = "lancet",
               sort.val = "asc",
               sort.by.groups = TRUE,
               x.text.angle = 90
) +
  ggtitle("10 Random Donner Members Age and Survival")
g

#P2 Data Preparation Step- Totaling the survivors
library(dplyr)
library(vcdExtra)

data(Donner)
print (Donner)
# Summarize data by family
Donner_summary <- Donner %>%
  group_by(family) %>%
  summarize(Total_Survivors = sum(survived == "1"),
            Family_Size = n()) %>%
  mutate(Percent_Survivors = Total_Survivors / Family_Size * 100)

print(Donner_summary)


#P3 pie chart of Donner Survivors by family

#Pie Chart

fam_survived = filter(Donner, survived == 1)
fam_surv_cnt= count (fam_survived, family)
# Note: df is fam_surv_cnt, col_1=family, col_2=n
ggpie(
  fam_surv_cnt, x = "n", label = "n",
  lab.pos = "in", lab.font = list(color = "white"),
  fill = "family", color = "white",
  palette = "jco"
)

#P4 Data Preparations for a bar chart of the survivors
#2x2 dataframe that summarizes the data found in the Donner data frame in the {vcdExtra} package
summary_df <- data.frame(
  Suvive = factor(c("survived", "died"), levels = c("survived", "died")),
  Number = c(sum(Donner$survived == "1"), sum(Donner$survived == "0"))
)
summary_df

#output 48 survived, 42 died

#P5
# Create data frame
summary_df <- data.frame(
  category = factor(c("survived", "died"), levels = c("survived", "died")),
  count = c(sum(Donner$survived == "1"), sum(Donner$survived == "0"))
)

# Create the bar plot using ggbarplot()
ggbarplot(summary_df, x = "category", y = "count",
          fill = "category",
          color = "black",
          palette = "jco",
          label = TRUE,
          ylab = "Number of survivors",
          ylim = c(0, 60), # Set the y-axis limit
          width = 0.7, # Set the width of the bars
          ggtheme = theme_pubclean()) + # Use a clean theme
  ggtitle("Donner People Survived and Died") # Add a title

#P6 bar chart age of donner party
# Load data
# Load data
data("Donner")
dfm <- Donner
# Convert the family variable to a factor
dfm$survived <- as.factor(dfm$survived)
# Add the name colums
dfm$name <- rownames(dfm)
# Inspect the data
head(dfm[, c("family", "age", "sex", "survived")],90)

ggbarplot(dfm, x = "name", y = "age",
          fill = "survived",               # change fill color by cyl
          color = "black",            # Set bar border colors to white
          palette = "jco",            # jco journal color palett. see ?ggpar
          sort.val = "desc",          # Sort the value in dscending order
          sort.by.groups = FALSE,     # Don't sort inside each group
          x.text.angle = 90           # Rotate vertically x axis texts
)

#P7 – Age of the Donner Party (Bar Chart) 
# Load data
data("Donner")
dfm <- Donner
# Convert the family variable to a factor
dfm$survived <- as.factor(dfm$survived)
# Add the name colums
dfm$name <- rownames(dfm)
# Inspect the data
head(dfm[, c("family", "age", "sex", "survived")],90)

ggbarplot(dfm, x = "name", y = "age",
          fill = "survived",               # change fill color by cyl
          color = "black",            # Set bar border colors to white
          palette = "jco",            # jco journal color palett. see ?ggpar
          sort.val = "desc",          # Sort the value in dscending order
          sort.by.groups = TRUE,     # Don't sort inside each group
          x.text.angle = 90           # Rotate vertically x axis texts
)

#P8 Which families enjoyed the largest percentage of survivors? (dot chart)

data(Donner)
#print (Donner)
# Summarize data by family
Donner_summary <- Donner %>%
  group_by(family) %>%
  summarize(Total_Survivors = sum(survived == "1"),
            Family_Size = n()) %>%
  mutate(Percent_Survivors = Total_Survivors / Family_Size * 100)

#print(Donner_summary)
ggplot(Donner_summary, aes(family, Percent_Survivors)) +
  geom_linerange(
    aes(x = reorder(family,Percent_Survivors), ymin = 0, ymax = Percent_Survivors), 
    color = "lightgray", size = 1.5)+
  geom_point(aes(color = family), size = 2)+
  ggpubr::color_palette("jco")+
  theme_pubclean()


#P9 z-score of survivors by families (dot chart)

data(Donner)
#print (Donner)
# Summarize data by family
data(Donner)
#print (Donner)
# Summarize data by family
dfm <- Donner %>%
  group_by(family) %>%
  summarize(Total_Survivors = sum(survived == "1"),
            Family_Size = n()) %>%
  mutate(Percent_Survivors = Total_Survivors / Family_Size * 100)

#print(Donner_summary)
dfm$sur_z <- (dfm$Percent_Survivors -mean(dfm$Percent_Survivors))/sd(dfm$Percent_Survivors)
dfm$sur_grp <- factor(ifelse(dfm$sur_z < 0, "low", "high"), levels = c("low", "high"))

ggplot(dfm, aes(family, sur_z)) + 
  geom_linerange(
    aes(x = reorder(family,sur_z), ymin = 0, ymax = sur_z), 
    color = "lightgray", size = 1.5)+
  geom_point(aes(color = sur_grp), size = 2)+
  ggpubr::color_palette("jco")+
  theme_pubclean()

#P10 Survivors by Families (Cleveland Plot)

data(Donner)
#print (Donner)

dfm <- Donner %>%
  group_by(family) %>%
  summarize(Total_Survivors = sum(survived == "1"), Family_Size = n()) %>%
  mutate(Percent_Survivors = Total_Survivors / Family_Size * 100)

# Calculate z-score and group
dfm$sur_z <- (dfm$Percent_Survivors - mean(dfm$Percent_Survivors)) / sd(dfm$Percent_Survivors)
dfm$sur_grp <- factor(ifelse(dfm$sur_z < 0, "low", "high"), levels = c("low", "high"))

ggdotchart(dfm, x = "family", y = "sur_z",
           color = "sur_grp",                  	# Color by survived groups
           palette = c("#00AFBB", "#E7B800"), 	# Custom color palette
           sorting = "descending",               	# Sort value in descending order
           rotate = TRUE,                        	# Rotate vertically
           dot.size = 2,                       	# Large dot size
           y.text.col = TRUE,                    	# Color y text by groups
           ggtheme = theme_pubr()                	# ggplot2 theme
)+
  theme_cleveland()   

  