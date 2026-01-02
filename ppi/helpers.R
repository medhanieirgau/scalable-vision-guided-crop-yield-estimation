library(boot)
library(grf)

# Check if x is a whole number
is.wholenumber <- function(x, tol = .Machine$double.eps^0.5) {
  return(abs(x - round(x)) < tol)  
}

# Returns a list of K indices dividing indices 1:n at random into
# K folds as equally sized as possible
get_fold_inds <- function(n, K, seed=NULL, type="foldid") {
  if (K == 1) {
    inds <- list()
    inds[[1]] <- 1:n
    if (type == "foldid") {
      return(rep(1, n))
    } else {
      return(inds)
    }
  }
  set.seed(seed)
  perm <- sample(n)
  cut_ind <- cut(1:n, breaks=K, labels=FALSE)
  if (type == "standard") {
    return(lapply(1:K, function(k) perm[which(cut_ind==k)]))
  } else if (type == "foldid") {
    return(cut_ind[perm])
  } else {
    stop("type must be one of 'standard' or 'foldid'")
  }
}

# Prediction control lists for different regression methods for learning rectifier
get_predict_control_list_from_reg_method <- function(reg_method) {
  return(switch(EXPR=reg_method,
                llf=list(ll.lambda=0.1),
                rf=list(),
                lm=NULL,
                glmnet=list(s="lambda.1se"),
                stop(paste("reg_method", reg_method, "not supported"))))
}


## Estimator functions
# Compute PPI estimators and CI (Algorithms 1 and 2 combined)
# Inputs:
#   - lab_df: data frame of labeled data
#   - unlab_df: data frame of unlabeled data
#   - cov_names: list of column names for covariates to adjust for, excluding photo model predictions
#   - mhat_name: column name containing photo model predictions
#   - R: number of bootstrap replicates for average yield CI's. NULL gives CLT-based intervals
#   - boot_ci_type: bootstrap CI computation method; must be one of the options supported by boot.ci(): "norm", "basic", "stud", "perc", "bca"
#   - alpha: significance level for CI
#   - lam: method to compute PPI coefficient lambda. NULL uses the PPI++ estimate.
# Output: Three-element list containing point estimate and CI endpoints
compute_ppi_estimator <- function(lab_df, unlab_df, cov_names, 
                                  mhat_name="mhat", R=NULL,
                                  boot_ci_type=NULL, alpha=0.05, lam=NULL) {
  
  # Verify names in lab_df and unlab_df
  if (!(all(c(mhat_name, cov_names, "y") %in% names(lab_df)))) {
    stop(paste("labeled data frame does not have all of these columns:", paste(mhat_name, "y", cov_names)))
  }
  if (!(all(c(mhat_name, cov_names) %in% names(unlab_df)))) {
    stop(paste("unlabeled data frame does not have all of these columns:", paste(mhat_name, cov_names)))
  }
  
  n <- nrow(lab_df)
  N <- nrow(unlab_df)
  
  # Add meaningless labels to unlabeled data
  unlab_df$y <- 0
  
  # Subset to relevant columns
  all_cols <- c(cov_names, mhat_name, "y")
  lab_df <- lab_df[,all_cols]
  unlab_df <- unlab_df[,all_cols]
  
  # Combine labeled and unlabeled data
  all_df <- rbind(lab_df, unlab_df) %>%
    mutate(s=c(rep(1, nrow(lab_df)), rep(0, nrow(unlab_df))))
  
  # Function to compute estimator and estimator variance estimate, 
  # in the format expected by boot package
  statistic <- function(data, i, lam, mhat_name) {
    data_boot <- data[i,]
    lab <- data_boot[data_boot$s==1,]
    unlab <- data_boot[data_boot$s==0,]
    n <- nrow(lab)
    N <- nrow(unlab)
    
    # Estimate multiplier lambda
    if (is.null(lam)) {
      lam_num <- as.numeric(cov(lab$y, lab[[mhat_name]]))
      lam_den <- (n+N) / N * var(c(lab[[mhat_name]], unlab[[mhat_name]]))
      lam <- ifelse(lam_den == 0, 0, lam_num / lam_den)
    }
    V_f <- (lam)^2 * var(data_boot[[mhat_name]])
    V_del <- var(lab$y - lam * lab[[mhat_name]])
    Sigma_hat <- V_f / N + V_del / n
    return(c(mean(lab$y) + lam * (mean(unlab[[mhat_name]]) - mean(lab[[mhat_name]])),
             Sigma_hat))
  } 
  
  # Compute estimator
  all_out <- statistic(data=all_df, i=1:nrow(all_df), lam=lam, mhat_name=mhat_name)
  theta_hat <- all_out[1]
  
  # Bootstrap CI's
  if (!is.null(R)) {
    if (!(boot_ci_type %in% c("normal", "basic", "student", "percent", "bca"))) {
      warning("boot_ci_type should be one of 'normal', 'basic', 'student', 
              'percent', or 'bca')")
    }
    
    # Stratified bootstrap resamples
    boot_out <- boot(data=all_df, statistic=statistic, R=R, strata=all_df$s,
                     lam=lam, mhat_name=mhat_name)
    
    if (var(boot_out$t[,1]) == 0) {
      lower <- theta_hat
      upper <- theta_hat
    } else {
      # Bootstrap CI's
      boot_ci_out <- boot.ci(boot.out=boot_out, conf=1-alpha, lam=lam, 
                             mhat_name=mhat_name)
      interval <- boot_ci_out[[boot_ci_type]]
      lower <- interval[length(interval)-1]
      upper <- interval[length(interval)] 
    }
    
    # Standard within-sample CI
  } else {
    se <- sqrt(all_out[2])
    lower <- theta_hat - qnorm(1-alpha/2) * se
    upper <- theta_hat + qnorm(1-alpha/2) * se
  }
  return(list(point_est=max(0, theta_hat), 
              ci_lower=max(0, lower), ci_upper=max(0, upper)))
}

# Helper function to train control function
# Inputs:
#   - reg_method: Learning algorithm to use. Currently supports:
#     - "llf": local linear forest (not shown in paper due to training time)
#     - "rf": generalized random forest
#     - "lm": ordinary least squares
#     - "glmnet": cross-validated LASSO (as described in main text)
#.  - X: numerical matrix of covariates (includes photo model predictions for our application, i.e., W)
#.  - Y: numerical vector of outcomes (yields in our case)
#.  - control_list: control_list for custom training options for algorithm
# Output: trained control function, as model object of appropriate class for reg_method
train_model <- function(reg_method, X, Y, control_list) {
  if (reg_method == "llf") {
    return(do.call(ll_regression_forest, c(list(X=X, Y=Y), control_list)))
  } else if (reg_method == "rf") {
    return(do.call(regression_forest, c(list(X=X, Y=Y), control_list)))
  } else if (reg_method == "lm") {
    lm_df <- as.data.frame(cbind(Y, X))
    names(lm_df) <- c("y", names(X))
    return(do.call(lm, c(list(formula=y ~ ., data=lm_df), control_list)))
  } else if (reg_method == "glmnet") {
    fit <- tryCatch(do.call(cv.glmnet, c(list(x=as.matrix(X), y=as.matrix(Y)), control_list)),
                    error=function(e) {
                      message(e)
                      warning("Error in cv.glmnet. Setting reg_method to 'lm'\n")
                      control_list <- NULL
                      lm_df <- as.data.frame(cbind(Y, X))
                      names(lm_df) <- c("y", names(X))
                      return(do.call(lm, c(list(formula=y ~ ., data=lm_df), control_list)))
                    })
    return(fit)
  } else {
    stop("reg_method not supported")
  }
}

# Helper function to evaluate trained control function
# Inputs:
#   - model: trained control function
#   - newdata: new matrix to evaluate control function on
#   - reg_method: Learning algorithm used (must match model)
#.  - control_list: control_list for custom prediction options
# Output: vector of predicted outputs from evaluating model on newdata
predict_model <- function(model, newdata, reg_method, control_list) {
  if (inherits(model, "lm")) {
    reg_method <- "lm"
    control_list <- NULL
  }
  if (reg_method == "glmnet") {
    return(do.call(predict, c(list(object=model, newx=as.matrix(newdata)), 
                              control_list)))
  }
  raw_preds <- do.call(predict, c(list(object=model, newdata=newdata), control_list))
  if (reg_method == "llf") {
    return(raw_preds$predictions)
  } else if (reg_method == "rf") {
    return(raw_preds$predictions)
  } else {
    return(raw_preds)
  }
}

## Wrapper function to train all control functions and add predictions in columns mhat and mhat_nophoto
learn_rectifier <- function(lab_df, unlab_df, cov_names, yhat, reg_method,
                            sort_column, cf_folds, cv_folds=NULL, seed=NULL) {
  
  ## Preallocations
  K_cf <- length(unique(cf_folds))
  lbl_preds <- vector("list", K_cf)
  unlbl_mhat_preds <- matrix(nrow=nrow(unlab_df), ncol=K_cf)
  unlbl_mhat_nophoto_preds <- matrix(nrow=nrow(unlab_df), ncol=K_cf)
  
  ## Cross-fitting
  for (k in 1:K_cf) {
    test_ind <- cf_folds == k
    if (K_cf == 1) {
      train_ind <- test_ind
    } else {
      train_ind <- cf_folds != k
    }
    train <- lab_df[train_ind,]
    test <- lab_df[test_ind,]
    reg_control_list <- switch(EXPR=reg_method, 
                               llf=list(seed=seed), 
                               rf=list(seed=seed),
                               lm=NULL,
                               glmnet=list(alpha=1, 
                                           foldid=cv_folds[train_ind], 
                                           grouped=FALSE),
                               stop(paste("reg_method", reg_method, 
                                          "not supported")))
    
    ## Learn regional rectifier
    mhat_model <- train_model(reg_method=reg_method, 
                              X=train[,c(cov_names, yhat)],
                              Y=train$y, control_list=reg_control_list)
    mhat_nophoto_model <- train_model(reg_method=reg_method, 
                                    X=train[,cov_names],
                                    Y=train$y, control_list=reg_control_list)
    
    ## Add predictions
    predict_control_list <- get_predict_control_list_from_reg_method(reg_method)
    lbl_preds[[k]] <- test %>%
      mutate(mhat=predict_model(model=mhat_model, 
                                newdata=test[,c(cov_names, yhat)],
                                reg_method=reg_method, 
                                control_list=predict_control_list),
             mhat_nophoto=predict_model(model=mhat_nophoto_model, 
                                      newdata=test[,cov_names],
                                      reg_method=reg_method, 
                                      control_list=predict_control_list))
    unlbl_mhat_preds[,k] <- predict_model(model=mhat_model, 
                                          newdata=unlab_df[,c(cov_names, yhat)],
                                          reg_method=reg_method, 
                                          control_list=predict_control_list)
    unlbl_mhat_nophoto_preds[,k] <- predict_model(model=mhat_nophoto_model,
                                                newdata=unlab_df[,cov_names],
                                                reg_method=reg_method,
                                                control_list=predict_control_list)
  }
  
  ## Combine info from different folds
  lab <- do.call(rbind, lbl_preds) %>%
    arrange(!!sym(sort_column))
  unlab <- unlab_df %>%
    mutate(mhat=rowMeans(unlbl_mhat_preds),
           mhat_nophoto=rowMeans(unlbl_mhat_nophoto_preds))
  return(list(lab=lab, unlab=unlab))
}

## Evaluation functions
# Estimate MSE
get_mse <- function(est_df, thetas, weighted=FALSE) {
  err_sq <- (est_df$point_est - thetas)^2
  if (!weighted) {
    return(mean(err_sq))
  } else {
    sizes <- est_df$size
    return(sum(err_sq*sizes) / sum(sizes))
  }
}

# Estimate squared bias 
get_sq_bias <- function(est_df, thetas) {
  return((mean(est_df$point_est - thetas))^2)
}

# Estimate CI coverage
get_coverage <- function(est_df, thetas) {
  return(mean(est_df$ci_lower <= thetas & est_df$ci_upper >= thetas))
}

# Estimate mean CI width, conditional on nonzero
get_mean_ci_widths <- function(est_df, weighted=FALSE) {
  widths <- est_df$ci_upper - est_df$ci_lower
  if (!weighted) {
    return(mean(widths))
  } else {
    sizes <- est_df$size
    return(sum(widths*sizes) / sum(sizes))
  }
}


# Aggregate sim results across study regions
aggregate_results_over_study_regions <- function(sim_outs) {
  estimators <- names(sim_outs[[1]])
  n_study_regions <- length(sim_outs)
  out <- lapply(estimators, function(estimator) do.call(rbind, lapply(1:n_study_regions, function(r) sim_outs[[r]][[estimator]])))
  names(out) <- estimators
  return(out)
}

# Function to run simulations for a single (country, year)
# Inputs:
#   - lbl_by_study_region: list of labeled data frames, broken down by study region
#   - unlbl_by-study_region: list of unlabeled data frames, broken down by study region
#   - cov_names: list of covariates to adjust for (excluding photo model predictions)
#   - yhat: name of column containing photo model predictions
#   - K_cf: number of cross-fitting folds for learning control function
#   - K_cv: number of cross-validation folds for learning control function
#   - R: number of bootstrap replicates for average yield CI's. NULL gives CLT-based intervals
#   - boot_ci_type: bootstrap CI computation method; must be one of the options supported by boot.ci(): "norm", "basic", "stud", "perc", "bca"
#   - alpha: significance level for CI
#   - rectifier_scale: scale of pooling for learning control function, or "rectifier"
#   - boot: whether to resample the data frames in lbl_by_study_region and unlbl_by_study_region (always TRUE for our simulations)
#   - boot_n_lbl_mult: if boot is TRUE, multiple of original zone sample size for each resampled labeled dataset. ignored if boot is FALSE.
#   - boot_n_unlbl_mult: if boot is TRUE, multiple of original zone sample size for each resampled unlabeled dataset. ignored if boot is FALSE.
#.  - seed: random seed
# Output: complex nested list of results, to be processed by the functions in figures.Rmd.
run_sims <- function(lbl_by_study_region, unlbl_by_study_region, cov_names, 
                     yhat, K_cf=1, K_cv=5, R=NULL, boot_ci_type=NULL,
                     alpha=0.05, rectifier_scale="zone", reg_method="glmnet", 
                     boot=FALSE, boot_n_lbl_mult=NULL, boot_n_unlbl_mult=NULL, 
                     seed=NULL) {
  
  if (!(rectifier_scale %in% c("zone", "study_region", "global"))) {
    stop("rectifier_scale must be one of 'zone', 'study_region,' or 'global'")
  }
  
  if (is.null(seed)) {
    seed <- sample(1e9, size=1)
  }
  
  ## lbl_by_study_region and unlbl_by_study_region must have same study regions
  if (!all(names(lbl_by_study_region) == names(unlbl_by_study_region))) {
    stop("lbl_by_study_region and unlbl_by_study_region do not have same study regions")
  }
  study_region_names <- names(lbl_by_study_region)
  n_study_regions <- length(study_region_names)
  
  ## Warn if yhat name is a covariate
  if (!is.null(yhat) && yhat %in% cov_names) {
    warning(paste("Column", yhat, "specified as both covariate and yhat, removing from cov_names"))
    cov_names <- cov_names[cov_names != yhat]
  }
  
  ## Further split data by zone
  lbl_by_study_region_by_zone <- lapply(lbl_by_study_region, function(df) split(df, f=df$zone))
  zone_sizes <- lapply(lbl_by_study_region_by_zone, function(dfs) sapply)
  unlbl_by_study_region_by_zone <- lapply(unlbl_by_study_region, function(df) split(df, f=df$zone))
  
  ## Bootstrap
  if (boot) {
    set.seed(seed)
    
    ## boot_n is NULL means full bootstrap
    if (is.null(boot_n_lbl_mult)) {
      boot_n_lbl_mult <- 1.0001
    }
    if (is.null(boot_n_unlbl_mult)) { 
      boot_n_unlbl_mult <- 1.0001
    }
    
    ## Number of times to resample each sample
    lbl_boot_times_by_zone <- lapply(lbl_by_study_region_by_zone, function(zone_dfs) {
      lapply(zone_dfs, function(df) {
        as.vector(rmultinom(n=1, size=ifelse(is.wholenumber(boot_n_lbl_mult), 
                                             boot_n_lbl_mult, 
                                             round(boot_n_lbl_mult*nrow(df))), 
                            prob=rep(1, nrow(df))))
      })
    })
    unlbl_boot_times_by_zone <- lapply(unlbl_by_study_region_by_zone, function(zone_dfs) {
      lapply(zone_dfs, function(df) {
        as.vector(rmultinom(n=1, size=ifelse(is.wholenumber(boot_n_unlbl_mult), 
                                             boot_n_unlbl_mult, round(boot_n_unlbl_mult*nrow(df))), 
                            prob=rep(1, nrow(df))))
      })
    })
    
    ## Assign random integer 1-K_cv to each labeled observation, for glmnet folds
    rcv_lbl_by_study_region <- lapply(1:n_study_regions, function(r) {
      lapply(1:length(lbl_by_study_region_by_zone[[r]]), function(z) {
        rep(sample(K_cv, size=nrow(lbl_by_study_region_by_zone[[r]][[z]]), replace=TRUE), 
            lbl_boot_times_by_zone[[r]][[z]])
      })
    })
    
    ## Assign random integer 1-K_cf to each labeled observation, for cross-fitting folds
    rcf_lbl_by_study_region <- lapply(1:n_study_regions, function(r) {
      lapply(1:length(lbl_by_study_region_by_zone[[r]]), function(z) {
        rep(sample(K_cf, size=nrow(lbl_by_study_region_by_zone[[r]][[z]]), replace=TRUE), 
            lbl_boot_times_by_zone[[r]][[z]])
      })
    })
    
    ## Perform resampling
    lbl_by_study_region_by_zone <- lapply(1:n_study_regions, function(r) {
      lapply(1:length(lbl_by_study_region_by_zone[[r]]), function(z) {
        curr_lbl <- lbl_by_study_region_by_zone[[r]][[z]]
        return(curr_lbl[rep(1:nrow(curr_lbl), lbl_boot_times_by_zone[[r]][[z]]),] %>%
                 mutate(cv_fold=rcv_lbl_by_study_region[[r]][[z]],
                        cf_fold=rcf_lbl_by_study_region[[r]][[z]]))
      })
    })
    unlbl_by_study_region_by_zone <- lapply(1:n_study_regions, function(r) {
      lapply(1:length(unlbl_by_study_region_by_zone[[r]]), function(z) {
        curr_unlbl <- unlbl_by_study_region_by_zone[[r]][[z]]
        return(curr_unlbl[rep(1:nrow(curr_unlbl), unlbl_boot_times_by_zone[[r]][[z]]),])
      })
    })
    
    ## Reaggregate resampled data to zone level
    lbl_by_study_region <- lapply(lbl_by_study_region_by_zone, function(dfs) do.call(rbind, dfs))
    unlbl_by_study_region <- lapply(unlbl_by_study_region_by_zone, function(dfs) do.call(rbind, dfs))
  }
  
  ## Extract zone names
  lbl_zone_names <- lapply(lbl_by_study_region, function(df) unique(df$zone))
  unlbl_zone_names <- lapply(unlbl_by_study_region, function(df) unique(df$zone))
  
  ## Verify no zone appears in more than one region
  all_lbl_zone_names <- sort(do.call("c", lbl_zone_names))
  all_unlbl_zone_names <- sort(do.call("c", unlbl_zone_names))
  if(!(length(all_lbl_zone_names) == length(unique(all_lbl_zone_names)))) {
    stop("one or more zones appear in multiple study regions in the labeled data")
  }
  if(!(length(all_unlbl_zone_names) == length(unique(all_unlbl_zone_names)))) {
    stop("one or more zones appear in multiple study regions in the unlabeled data")
  }
  if(!all(all_lbl_zone_names == all_unlbl_zone_names)) {
    stop("labeled and unlabeled data do not contain the same zones")
  }
  
  ## Options for prediction based on reg_method
  predict_control_list <- get_predict_control_list_from_reg_method(reg_method)
  
  ## Combine data across study regions
  all_lbl <- do.call(rbind, lbl_by_study_region)
  all_lbl <- all_lbl %>%
    mutate(obs_ind=1:nrow(all_lbl))
  all_unlbl <- do.call(rbind, unlbl_by_study_region)
  all_unlbl <- all_unlbl %>%
    mutate(obs_ind=1:nrow(all_unlbl))
  
  ## Learn global rectifier
  if (rectifier_scale == "global") {
    message("Learning global rectifier")
    
    ## Get cross validation and cross fit folds
    if (boot) {
      cf_folds <- all_lbl$cf_fold
      cv_folds <- all_lbl$cv_fold
    } else {
      cf_folds <- get_fold_inds(nrow(all_lbl), K_cf, seed=seed, type="foldid")
      cv_folds <- get_fold_inds(nrow(all_lbl), K_cv, seed=seed+1, type="foldid")
    }
    cf_folds <- dense_rank(cf_folds)
    cv_folds <- dense_rank(cv_folds)
    rect_out <- learn_rectifier(lab_df=all_lbl, unlab_df=all_unlbl, 
                                cov_names=cov_names, yhat=yhat, 
                                reg_method=reg_method, 
                                sort_column="obs_ind", cf_folds=cf_folds, 
                                cv_folds=cv_folds, seed=seed)
    all_lbl <- rect_out$lab
    all_unlbl <- rect_out$unlab
  }
  
  ## Re-split observations by study region
  lbl_split_ind <- do.call("c", lapply(1:n_study_regions, function(i) rep(i, nrow(lbl_by_study_region[[i]]))))
  unlbl_split_ind <- do.call("c", lapply(1:n_study_regions, function(i) rep(i, nrow(unlbl_by_study_region[[i]]))))
  lbl_by_study_region <- split(all_lbl, lbl_split_ind)
  unlbl_by_study_region <- split(all_unlbl, unlbl_split_ind)
  names(lbl_by_study_region) <- study_region_names
  names(unlbl_by_study_region) <- study_region_names
  
  ## Loop over regions
  all_results <- vector("list", n_study_regions)
  z_alpha <- qnorm(1-alpha/2)
  for (r in 1:n_study_regions) {
    # message(paste0("Processing study region ", r, " of ", n_study_regions, ": ", 
    #                study_region_names[r]))
    study_region_lbl <- lbl_by_study_region[[r]]
    study_region_unlbl <- unlbl_by_study_region[[r]]
    
    ## Learn rectifier by study region
    if (rectifier_scale == "study_region") {
      if (boot) {
        cf_folds <- study_region_lbl$cf_fold
        cv_folds <- study_region_lbl$cv_fold
      } else {
        cf_folds <- get_fold_inds(nrow(study_region_lbl), K_cf, seed=seed, type="foldid")
        cv_folds <- get_fold_inds(nrow(study_region_lbl), K_cv, seed=seed+1, type="foldid")
      }
      cf_folds <- dense_rank(cf_folds)
      cv_folds <- dense_rank(cv_folds)
      rect_out <- learn_rectifier(lab_df=study_region_lbl, 
                                  unlab_df=study_region_unlbl, 
                                  cov_names=cov_names, yhat=yhat, 
                                  reg_method=reg_method, 
                                  sort_column="obs_ind", cf_folds=cf_folds, 
                                  cv_folds=cv_folds, seed=seed)
      study_region_lbl <- rect_out$lab
      study_region_unlbl <- rect_out$unlab
    }
    
    ## Split into zones
    grp <- nest(study_region_lbl, .by="zone") %>%
      left_join(nest(study_region_unlbl, .by="zone"), by="zone") %>%
      rename(lbl=data.x, unlbl=data.y)
    
    ## Preallocations
    n_zones <- nrow(grp)
    blank_df <- data.frame(point_est=rep(NA, n_zones),
                           ci_lower=rep(NA, n_zones),
                           ci_upper=rep(NA, n_zones),
                           row.names=grp$zone)
    lbl <- blank_df
    aipw <- blank_df
    ppi <- blank_df
    ppipp <- blank_df
    nophoto <- blank_df
    
    ## Loop over zones
    for (i in 1:n_zones) {
      ## Extract data for zone
      lab <- grp$lbl[[i]]
      unlab <- grp$unlbl[[i]]
      n <- nrow(lab)
      N <- nrow(unlab)
      
      ## Learn rectifier by zone
      if (rectifier_scale == "zone") {
        if (boot) {
          cf_folds <- lab$cf_fold
          cv_folds <- lab$cv_fold
        } else {
          cf_folds <- get_fold_inds(nrow(lab), K_cf, seed=seed, type="foldid")
          cv_folds <- get_fold_inds(nrow(lab), K_cv, seed=seed+1, type="foldid")
        }
        cf_folds <- dense_rank(cf_folds)
        cv_folds <- dense_rank(cv_folds)
        rect_out <- learn_rectifier(lab_df=lab, unlab_df=unlab, 
                                    cov_names=cov_names, yhat=yhat, 
                                    reg_method=reg_method, 
                                    sort_column="obs_ind", cf_folds=cf_folds, 
                                    cv_folds=cv_folds, seed=seed)
        lab <- rect_out$lab
        unlab <- rect_out$unlab
      }

      ## Compute estimators
      ## lbl
      lbl[i,] <- do.call("c", compute_ppi_estimator(lab_df=lab,
                                                    unlab_df=unlab,
                                                    cov_names=cov_names,
                                                    mhat_name="mhat_nophoto",
                                                    R=R, 
                                                    boot_ci_type=boot_ci_type,
                                                    alpha=alpha, lam=0))
      
      ## nophoto
      nophoto[i,] <- do.call("c", compute_ppi_estimator(lab_df=lab,
                                                      unlab_df=unlab,
                                                      cov_names=cov_names,
                                                      mhat_name="mhat_nophoto",
                                                      R=R, 
                                                      boot_ci_type=boot_ci_type, 
                                                      alpha=alpha, lam=NULL))
      
      ## AIPW, PPI, and PPI++
      aipw[i, ] <- do.call("c", compute_ppi_estimator(lab_df=lab, 
                                                      unlab_df=unlab, 
                                                      cov_names=c(cov_names, yhat), 
                                                      mhat_name="mhat",
                                                      R=R, 
                                                      boot_ci_type=boot_ci_type,
                                                      alpha=alpha, lam=N/(n+N)))
      
      ppi[i, ] <- do.call("c", compute_ppi_estimator(lab_df=lab, 
                                                     unlab_df=unlab, 
                                                     cov_names=c(cov_names, yhat), 
                                                     mhat_name="mhat",
                                                     R=R, 
                                                     boot_ci_type=boot_ci_type,
                                                     alpha=alpha, lam=1))
      ppipp[i, ] <- do.call("c", compute_ppi_estimator(lab_df=lab, 
                                                       unlab_df=unlab, 
                                                       cov_names=c(cov_names, yhat), 
                                                       mhat_name="mhat",
                                                       R=R, 
                                                       boot_ci_type=boot_ci_type,
                                                       alpha=alpha, lam=NULL))
    }
    ## Store result
    all_results[[r]] <- list(lbl=lbl, aipw=aipw, ppi=ppi, ppipp=ppipp, nophoto=nophoto)
    
    ## Increment seed
    seed <- seed + 1
  }
  names(all_results) <- study_region_names
  return(all_results)
}
