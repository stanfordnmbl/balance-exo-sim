require("lme4")
require("multcomp")
require("broom")
require("emmeans")

citation("lme4")
citation("multcomp")
citation("broom")
citation("emmeans")
citation("lmerTest")

verbosity = 1

# Redirect all output to a text file.
sink("center_of_mass_statistics.txt", append=FALSE, split=TRUE)

perform_stats_for_formula <- function(exp, formula, linfct, result_path) {
    print("");
    print("");
    print("=========================== formula: =============================")
    print(formula)
    print("==================================================================")
    model = lmerTest::lmer(formula, data=exp.data)
    if (verbosity == 1) {
        print(summary(model))
        print(confint(model))
    }
    z <- anova(model)
    print(z)

    # Perform pairwise comparisons.
    comparisons = multcomp::glht(model, linfct=linfct)
    comparisons_path <- paste0(result_path, "_comparisons.csv")
    write.csv(tidy(comparisons), file=comparisons_path)

    if (verbosity == 1) {
        print(summary(comparisons,
                      ))
    }
}

# Did all of the perturbations change the center-of-mass kinematics?
actuators <- c("muscles", "torques")
times <- c("20", "25", "30", "35", "40", "45", "50", "55", "60")
kins <- c("pos", "vel", "acc")
direcs <- c("x", "y", "z")
for (actuator in actuators) {
    for (time in times) {
        for (kin in kins) {
            for (direc in direcs) {
                filepath <- paste0("tables/com_stats_time", time, "_", kin, "_", direc, "_", actuator, ".csv")
                result_fpath <- paste0("results/com_stats_time", time, "_", kin, "_", direc, "_", actuator)

                print("");
                print("");
                print("=========================== filename: ============================")
                print(filepath)
                print("==================================================================")
                exp.data = read.csv(filepath, fileEncoding="UTF-8-BOM")
                exp.data$perturbation <- as.factor(exp.data$perturbation)
                exp.data$subject <- as.factor(exp.data$subject)
                str(exp.data)

                # Make "unperturbed" the reference level for the perturbation factor.
                exp.data <- within(exp.data, perturbation <- relevel(perturbation, ref="unperturbed"))

                # Run the stats.
                perform_stats_for_formula(exp, com ~ perturbation + (1 | subject), 
                    emm(pairwise ~ perturbation, parens=NULL), result_fpath)
            }
        }
    }
}

# Were the torque-driven perturbations different from the muscle-driven perturbations?
times <- c("20", "25", "30", "35", "40", "45", "50", "55", "60")
kins <- c("pos", "vel", "acc")
direcs <- c("x", "y", "z")
for (time in times) {
    for (kin in kins) {
        for (direc in direcs) {
            filepath <- paste0("tables/com_stats_time", time, "_", kin, "_", direc, "_diff.csv")
            result_fpath <- paste0("results/com_stats_time", time, "_", kin, "_", direc, "_diff")

            print("");
            print("");
            print("=========================== filename: ============================")
            print(filepath)
            print("==================================================================")
            exp.data = read.csv(filepath, fileEncoding="UTF-8-BOM")
            exp.data$perturbation <- as.factor(exp.data$perturbation)
            exp.data$subject <- as.factor(exp.data$subject)
            exp.data$actuator <- as.factor(exp.data$actuator)
            str(exp.data)

            # Run the stats.
            perform_stats_for_formula(exp, com ~ actuator + perturbation + (1 | subject), 
                    emm(pairwise ~ perturbation * actuator, parens=NULL), result_fpath)
        }
    }
}


sink()