'''lightGBM'''

library(caret)
library(randomForest)
library(openxlsx)
library(rlist)
library(dplyr)
library(foreach)
library(doParallel)
library(dplyr)
library(mlr)
library(parallelMap)
library(lightgbm)
library(readr)
library(ggplot2)
library(ggvis)
library(pROC)


filepath = 'E:\\niewei\\Desktop\\2018工作日志\\201805\\外部数据\\探知\\探知lightGBM'
setwd(filepath)
origin_data = read.csv('data_model.csv')
origin_data = origin_data[complete.cases(origin_data$yq_1),]
#提取20180425之前的数据集
dataSet = with(origin_data,origin_data[add_time <= 20180425,])

#T+5
dataSet = mutate(dataSet,'target' = ifelse(yq_1 > 5,1,0))
#dataSet$target = as.factor(dataSet$target)

#split data
data_OTT = with(dataSet,dataSet[add_time > 20180423,])[14:79] %>% within(.,rm(yq_1))
data_model = with(dataSet,dataSet[add_time <= 20180423,])[14:79] %>% within(.,rm(yq_1))
cat('时间外样本量：',dim(data_OTT)[1],'\n','bad rate：',prop.table(table(data_OTT$target))[2] %>% round(.,3))
cat('训练集样本量：',dim(data_model)[1],'\n','bad rate：',prop.table(table(data_model$target))[2] %>% round(.,3))


set.seed(888)
Index = createDataPartition(data_model$target,
                            times = 1,p = .7,list = FALSE)
lgb_tr1 = data_model[Index,] 
lgb_te1  = data_model[-Index,] 


#parallelStop()
parallelStartSocket(2)

#summarizeColumns(lgb_tr2) %>% View()
imp_tr1 <- impute(
  as.data.frame(lgb_tr1), 
  classes = list(
    integer = imputeMedian(), 
    numeric = imputeMedian()
  )
)
imp_te1 <- impute(
  as.data.frame(lgb_te1), 
  classes = list(
    integer = imputeMedian(), 
    numeric = imputeMedian()
  )
)

#处理缺失值之后的train_data和test_data
lgb_tr2 = imp_tr1$data
lgb_te2 = imp_te1$data


"""特征选取"""
lgb_tr2$target <- factor(lgb_tr2$target)
lgb.task <- makeClassifTask(data = lgb_tr2, target = 'target')
lgb.task.smote <- oversample(lgb.task, rate = 5)
fv_time <- system.time(
  fv <- generateFilterValuesData(
    lgb.task.smote,
    method = c('chi.squared') ## 信息增益/卡方检验的方法
  )
)

plotFilterValues(fv)
#plotFilterValuesGGVIS(fv)
fv_data2 <- fv$data %>%
  arrange(desc(chi.squared)) %>%
  mutate(chi_gain_cul = cumsum(chi.squared) / sum(chi.squared))
fv_data2_filter <- fv_data2 %>% filter(chi_gain_cul <= 0.99)
dim(fv_data2_filter)
fv_feature <- fv_data2_filter$name
lgb_tr3 <- lgb_tr2[, c(fv_feature, 'target')]
lgb_te3 <- lgb_te2[, c(fv_feature,'target')]

#保存数据集
write_csv(lgb_tr3, 'lgb_tr3_chi.csv')
write_csv(lgb_te3, 'lgb_te3_chi.csv')




'''调整参数'''
lgb_tr3$target = as.numeric(as.character(lgb_tr3$target))
lgb_te3$target = as.numeric(as.character(lgb_te3$target))

# lgb_tr2$target = as.numeric(as.character(lgb_tr2$target))
# lgb_te2$target = as.numeric(as.character(lgb_te2$target))


table(lgb_tr3$target)[1] / table(lgb_tr3$target)[2]
grid_search <- expand.grid(
  weight = seq(1, 20, 2) 
  ## 故而设定weight在[1, 20]之间
)
lgb_rate_1 <- numeric(length = nrow(grid_search))


"""调整weight参数"""
for(i in 1:nrow(grid_search)){
  lgb_weight <- (lgb_tr3$target * i + 1) / sum(lgb_tr3$target * i + 1)
  lgb_train <- lgb.Dataset(
    data = data.matrix(lgb_tr3[, 1:(dim(lgb_tr3)[2] - 1)]), 
    label = lgb_tr3$target, 
    free_raw_data = FALSE,
    weight = lgb_weight
  )
  # 参数
  params <- list(
    objective = 'binary',
    metric = 'auc'
  )
  # 交叉验证
  lgb_tr2_mod <- lgb.cv(
    params,
    data = lgb_train,  #训练矩阵
    nfold = 5,        #十折交叉
    learning_rate = 0.1,   #学习率
    early_stopping_rounds = 10   
  )
  lgb_rate_1[i] <- unlist(lgb_tr2_mod$record_evals$valid$auc$eval)[length(unlist(lgb_tr2_mod$record_evals$valid$auc$eval))]
}

grid_search$perf <- lgb_rate_1
ggplot(grid_search,aes(x = weight, y = perf)) + 
  geom_line()

#可以看出权重weight = 7的时候auc最大

"""调整learn_rate参数"""
grid_search <- expand.grid(
  learning_rate = 2 ^ (-(8:0)))
perf_learning_rate_1 <- numeric(length = nrow(grid_search))

for(i in 1:nrow(grid_search)){
  lgb_weight <- (lgb_tr3$target * 4 + 1) / sum(lgb_tr3$target * 4 + 1)
  lgb_train <- lgb.Dataset(
    data = data.matrix(lgb_tr3[, 1:dim(lgb_tr3)[2] - 1]), 
    label = lgb_tr3$target, 
    free_raw_data = FALSE,
    weight = lgb_weight
  )
  # 参数
  params <- list(
    objective = 'binary',
    metric = 'auc',
    learning_rate = grid_search[i,'learning_rate']
  )
  # 交叉验证
  lgb_tr_mod <- lgb.cv(
    params,
    data = lgb_train,
    nfold = 5,
    early_stopping_rounds = 10
  )
  perf_learning_rate_1[i] <- unlist(lgb_tr_mod$record_evals$valid$auc$eval)[length(unlist(lgb_tr_mod$record_evals$valid$auc$eval))]
}

grid_search$perf <- perf_learning_rate_1
ggplot(grid_search,aes(x = learning_rate, y = perf)) + 
  geom_point() +
  geom_smooth()


""".调整num_leaves参数"""

grid_search <- expand.grid(
  learning_rate = .125,
  num_leaves = seq(50, 800, 50)
)
perf_num_leaves_1 <- numeric(length = nrow(grid_search))

for(i in 1:nrow(grid_search)){
  lgb_weight <- (lgb_tr3$target * 4 + 1) / sum(lgb_tr3$target * 4 + 1)
  lgb_train <- lgb.Dataset(
    data = data.matrix(lgb_tr3[, 1:dim(lgb_tr3)[2] -  1]), 
    label = lgb_tr3$target, 
    free_raw_data = FALSE,
    weight = lgb_weight
  )


  # 参数
  params <- list(
    objective = 'binary',
    metric = 'auc',
    learning_rate = grid_search[i, 'learning_rate'],
    num_leaves = grid_search[i, 'num_leaves']
  )
  # 交叉验证
  lgb_tr_mod <- lgb.cv(
    params,
    data = lgb_train,
    nfold = 10,
    early_stopping_rounds = 10
  )
  perf_num_leaves_1[i] <- unlist(lgb_tr_mod$record_evals$valid$auc$eval)[length(unlist(lgb_tr_mod$record_evals$valid$auc$eval))]
}

grid_search$perf <- perf_num_leaves_1
ggplot(grid_search,aes(x = num_leaves, y = perf)) + 
  geom_point() +
  geom_smooth()



'''调试min_data_in_leaf参数'''
# grid_search <- expand.grid(
#   learning_rate = .7,
#   num_leaves = 300,
#   min_data_in_leaf = 2 ^ (1:7)
# )
# 
# perf_min_data_in_leaf_1 <- numeric(length = nrow(grid_search))
# 
# for(i in 1:nrow(grid_search)){
#   lgb_weight <- (lgb_tr3$target * 7 + 1) / sum(lgb_tr3$target * 7 + 1)
#   
#   lgb_train <- lgb.Dataset(
#     data = data.matrix(lgb_tr3[, 1:dim(lgb_tr3)[2] -  1]), 
#     label = lgb_tr3$target, 
#     free_raw_data = FALSE,
#     weight = lgb_weight
#   )
#   
#   # 参数
#   params <- list(
#     objective = 'binary',
#     metric = 'auc',
#     learning_rate = grid_search[i, 'learning_rate'],
#     num_leaves = grid_search[i, 'num_leaves'],
#     min_data_in_leaf = grid_search[i, 'min_data_in_leaf']
#   )
#   # 交叉验证
#   lgb_tr_mod <- lgb.cv(
#     params,
#     data = lgb_train,
#     nrounds = 300,
#     stratified = TRUE,
#     nfold = 5,
#     num_threads = 2,
#     early_stopping_rounds = 10
#   )
#   perf_min_data_in_leaf_1[i] <- unlist(lgb_tr_mod$record_evals$valid$auc$eval)[length(unlist(lgb_tr_mod$record_evals$valid$auc$eval))]
# }
# 
# grid_search$perf <- perf_min_data_in_leaf_1
# ggplot(grid_search,aes(x = min_data_in_leaf, y = perf)) + 
#   geom_point() +
#   geom_smooth()



'''max_bin参数'''
grid_search <- expand.grid(
  learning_rate = .7,
  num_leaves = 300,
  min_data_in_leaf = 20,
  max_bin = 2 ^ (4:10)
)

perf_max_bin_1 <- numeric(length = nrow(grid_search))

for(i in 1:nrow(grid_search)){
  lgb_weight <- (lgb_tr3$target * 7 + 1) / sum(lgb_tr3$target * 7 + 1)
  
  lgb_train <- lgb.Dataset(
    data = data.matrix(lgb_tr3[, 1:dim(lgb_tr3)[2] - 1]), 
    label = lgb_tr3$target, 
    free_raw_data = FALSE,
    weight = lgb_weight
  )
  
  # 参数
  params <- list(
    objective = 'binary',
    metric = 'auc',
    learning_rate = grid_search[i, 'learning_rate'],
    num_leaves = grid_search[i, 'num_leaves'],
    max_bin = grid_search[i, 'max_bin'],
    min_data_in_leaf = grid_search[i,'min_data_in_leaf']
  )
  # 交叉验证
  lgb_tr_mod <- lgb.cv(
    params,
    data = lgb_train,
    nrounds = 300,
    stratified = TRUE,
    nfold = 5,
    num_threads = 2,
    early_stopping_rounds = 10
  )
  perf_max_bin_1[i] <- unlist(lgb_tr_mod$record_evals$valid$auc$eval)[length(unlist(lgb_tr_mod$record_evals$valid$auc$eval))]
}

grid_search$perf <- perf_max_bin_1
ggplot(grid_search,aes(x = max_bin, y = perf)) + 
  geom_point() +
  geom_smooth()

#AUC最大max_bin设定为250


'''lambda参数调整'''
grid_search <- expand.grid(
  learning_rate = .7,
  num_leaves = 300,
  min_data_in_leaf = 20,
  max_bin=250,
  lambda_l1 = seq(0, .01, .002),
  lambda_l2 = seq(0, .01, .002)
)

perf_lamda_1 <- numeric(length = nrow(grid_search))

for(i in 1:nrow(grid_search)){
  lgb_weight <- (lgb_tr3$target * 7 + 1) / sum(lgb_tr3$target * 7 + 1)
  
  lgb_train <- lgb.Dataset(
    data = data.matrix(lgb_tr3[, 1:dim(lgb_tr3)[2] - 1]), 
    label = lgb_tr3$target, 
    free_raw_data = FALSE,
    weight = lgb_weight
  )
  
  # 参数
  params <- list(
    objective = 'binary',
    metric = 'auc',
    learning_rate = grid_search[i, 'learning_rate'],
    num_leaves = grid_search[i, 'num_leaves'],
    max_bin = grid_search[i, 'max_bin'],
    lambda_l1 = grid_search[i, 'lambda_l1'],
    lambda_l2 = grid_search[i, 'lambda_l2']
  )
  # 交叉验证
  lgb_tr_mod <- lgb.cv(
    params,
    data = lgb_train,
    nrounds = 300,
    stratified = TRUE,
    nfold = 5,
    num_threads = 2,
    early_stopping_rounds = 10
  )
  perf_lamda_1[i] <- unlist(lgb_tr_mod$record_evals$valid$auc$eval)[length(unlist(lgb_tr_mod$record_evals$valid$auc$eval))]
}

grid_search$perf <- perf_lamda_1
ggplot(data = grid_search, aes(x = lambda_l1, y = perf)) + 
  geom_point() + 
  facet_wrap(~ lambda_l2, nrow = 5)
#lambda 正则L1取0.0008,L2取0.0004


'''调试drop_rate参数'''
grid_search <- expand.grid(
  learning_rate = .7,
  num_leaves = 300,
  max_bin=250,
  lambda_l1 = 0.0008,
  lambda_l2 = 0.0004,
  drop_rate = seq(0, 1, .1)
)
perf_drop_rate_1 <- numeric(length = nrow(grid_search))
for(i in 1:nrow(grid_search)){
  lgb_weight <- (lgb_tr3$target * 7 + 1) / sum(lgb_tr3$target * 7 + 1)
  
  lgb_train <- lgb.Dataset(
    data = data.matrix(lgb_tr3[, 1:dim(lgb_tr3)[2] - 1]), 
    label = lgb_tr3$target, 
    free_raw_data = FALSE,
    weight = lgb_weight
  )
  
  # 参数
  params <- list(
    objective = 'binary',
    metric = 'auc',
    learning_rate = grid_search[i, 'learning_rate'],
    num_leaves = grid_search[i, 'num_leaves'],
    max_bin = grid_search[i, 'max_bin'],
    lambda_l1 = grid_search[i, 'lambda_l1'],
    lambda_l2 = grid_search[i, 'lambda_l2'],
    drop_rate = grid_search[i, 'drop_rate']
  )
  # 交叉验证
  lgb_tr_mod <- lgb.cv(
    params,
    data = lgb_train,
    nrounds = 300,
    stratified = TRUE,
    nfold = 5,
    num_threads = 2,
    early_stopping_rounds = 10
  )
  perf_drop_rate_1[i] <- unlist(lgb_tr_mod$record_evals$valid$auc$eval)[length(unlist(lgb_tr_mod$record_evals$valid$auc$eval))]
}

grid_search$perf <- perf_drop_rate_1
ggplot(data = grid_search, aes(x = drop_rate, y = perf)) + 
  geom_point() +
  geom_smooth()



'''调试max_drop参数'''
grid_search <- expand.grid(
  learning_rate = .7,
  num_leaves = 300,
  max_bin=250,
  lambda_l1 = 0.008,
  lambda_l2 = 0.004,
  drop_rate = .1,
  max_drop = seq(1, 10, 2)
)

perf_max_drop_1 <- numeric(length = nrow(grid_search))

for(i in 1:nrow(grid_search)){
  lgb_weight <- (lgb_tr3$target * 7 + 1) / sum(lgb_tr3$target * 7 + 1)
  
  lgb_train <- lgb.Dataset(
    data = data.matrix(lgb_tr3[, 1:dim(lgb_tr3)[2] - 1]), 
    label = lgb_tr3$target, 
    free_raw_data = FALSE,
    weight = lgb_weight
  )
  
  # 参数
  params <- list(
    objective = 'binary',
    metric = 'auc',
    learning_rate = grid_search[i, 'learning_rate'],
    num_leaves = grid_search[i, 'num_leaves'],
    max_bin = grid_search[i, 'max_bin'],
    lambda_l1 = grid_search[i, 'lambda_l1'],
    lambda_l2 = grid_search[i, 'lambda_l2'],
    drop_rate = grid_search[i, 'drop_rate'],
    max_drop = grid_search[i, 'max_drop']
  )
  # 交叉验证
  lgb_tr_mod <- lgb.cv(
    params,
    data = lgb_train,
    nrounds = 300,
    stratified = TRUE,
    nfold = 5,
    num_threads = 2,
    early_stopping_rounds = 10
  )
  perf_max_drop_1[i] <- unlist(lgb_tr_mod$record_evals$valid$auc$eval)[length(unlist(lgb_tr_mod$record_evals$valid$auc$eval))]
}

grid_search$perf <- perf_max_drop_1
ggplot(data = grid_search, aes(x = max_drop, y = perf)) + 
  geom_point() +
  geom_smooth()



"""测试集预测模型"""
#权重
lgb_weight <- (lgb_tr3$TARGET * 7 + 1) / sum(lgb_tr3$TARGET * 7 + 1)
#训练数据
lgb_train <- lgb.Dataset(
  data = data.matrix(lgb_tr3[,1:dim(lgb_tr3)[2] - 1]), 
  label = lgb_tr3$target, 
  free_raw_data = FALSE,
  weight = lgb_weight
)
#参数
params <- list(
  objective = 'binary',
  metric  = 'binary_logloss,auc',
  learning_rate = .7,
  num_leaves = 300,
  max_bin = 250,
  lambda_l1 = .0008,
  lambda_l2 = .0006
)
lgb_mod <- lightgbm(
  params = params,
  data = lgb_train,
  nrounds = 300,
  early_stopping_rounds = 10,
  num_threads = 2
)

'''测试集预测'''
lgb.pred <- predict(lgb_mod, data.matrix(lgb_te3))
model_roc = roc(lgb_te3$target,lgb.pred)
plot(model_roc, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"), max.auc.polygon=TRUE,
     auc.polygon.col="skyblue", print.thres=TRUE)


'''时间外样本测试'''
lgb.pred.OTT <- predict(lgb_mod, data.matrix(data_OTT))
OTT_roc = roc(data_OTT$target,lgb.pred.OTT)
plot(OTT_roc, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"), max.auc.polygon=TRUE,
     auc.polygon.col="skyblue", print.thres=TRUE)

'''测算KS'''
myKS <- function(pre,label){  
  true <- sum(label)  
  false <- length(label)-true  
  tpr <- NULL  
  fpr <- NULL  
  o_pre <- pre[order(pre)] # let the threshold in an order from small to large  
  for (i in o_pre){  
    tp <- sum((pre >= i) & label)  
    tpr <- c(tpr,tp/true)  
    fp <- sum((pre >= i) & (1-label))  
    fpr <- c(fpr,fp/false)  
  }  
  # plot(o_pre,tpr,type = "l",col= "green",xlab="threshold",ylab="tpr,fpr")  
  # lines(o_pre,fpr,type="l", col = "red")  
  KSvalue <- max(tpr-fpr)  
  # sub = paste("KS value =",KSvalue)  
  # title(sub=sub)  
  # cutpoint <- which(tpr-fpr==KSvalue)  
  # thre <- o_pre[cutpoint]  
  #lines(c(thre,thre),c(fpr[cutpoint],tpr[cutpoint]),col = "blue")  
  return(KSvalue)
}  
#'''测试集ks,时间外样本KS'''
cat('测试集ks:',myKS(lgb_te3$target,lgb.pred),'\n',
    '时间外样本ks:',myKS(data_OTT$target,lgb.pred.OTT))


# '''绘制KS曲线'''
# PlotKS_N<-function(Pred_Var, labels_Var, descending, N){  
#   # Pred_Var is prop: descending=1  
#   # Pred_Var is score: descending=0  
#   library(dplyr)  
#   df<- data.frame(Pred=Pred_Var, labels=labels_Var)  
#   if (descending==1){  
#     df1<-arrange(df, desc(Pred), labels)  
#   }else if (descending==0){  
#     df1<-arrange(df, Pred, labels)  
#   }  
#   df1$good1<-ifelse(df1$labels==0,1,0)  
#   df1$bad1<-ifelse(df1$labels==1,1,0)  
#   df1$cum_good1<-cumsum(df1$good1)  
#   df1$cum_bad1<-cumsum(df1$bad1)  
#   df1$rate_good1<-df1$cum_good1/sum(df1$good1)  
#   df1$rate_bad1<-df1$cum_bad1/sum(df1$bad1)  
#   
#   if (descending==1){  
#     df2<-arrange(df, desc(Pred), desc(labels))  
#   }else if (descending==0){  
#     df2<-arrange(df, Pred, desc(labels))  
#   }  
#   
#   df2$good2<-ifelse(df2$labels==0,1,0)  
#   df2$bad2<-ifelse(df2$labels==1,1,0)  
#   df2$cum_good2<-cumsum(df2$good2)  
#   df2$cum_bad2<-cumsum(df2$bad2)  
#   df2$rate_good2<-df2$cum_good2/sum(df2$good2)  
#   df2$rate_bad2<-df2$cum_bad2/sum(df2$bad2)  
#   
#   rate_good<-(df1$rate_good1+df2$rate_good2)/2  
#   rate_bad<-(df1$rate_bad1+df2$rate_bad2)/2  
#   df_ks<-data.frame(rate_good,rate_bad)  
#   
#   df_ks$KS<-df_ks$rate_bad-df_ks$rate_good  
#   
#   L<- nrow(df_ks)  
#   if (N>L) N<- L  
#   df_ks$tile<- 1:L  
#   qus<- quantile(1:L, probs = seq(0,1, 1/N))[-1]  
#   qus<- ceiling(qus)  
#   df_ks<- df_ks[df_ks$tile%in%qus,]  
#   df_ks$tile<- df_ks$tile/L  
#   df_0<-data.frame(rate_good=0,rate_bad=0,KS=0,tile=0)  
#   df_ks<-rbind(df_0, df_ks)  
#   
#   M_KS<-max(df_ks$KS)  
#   Pop<-df_ks$tile[which(df_ks$KS==M_KS)]  
#   M_good<-df_ks$rate_good[which(df_ks$KS==M_KS)]  
#   M_bad<-df_ks$rate_bad[which(df_ks$KS==M_KS)]  
#   
#   library(ggplot2)  
#   PlotKS<-ggplot(df_ks)+  
#     geom_line(aes(tile,rate_bad),colour="red2",size=1.2)+  
#     geom_line(aes(tile,rate_good),colour="blue3",size=1.2)+  
#     geom_line(aes(tile,KS),colour="forestgreen",size=1.2)+  
#     
#     geom_vline(xintercept=Pop,linetype=2,colour="gray",size=0.6)+  
#     geom_hline(yintercept=M_KS,linetype=2,colour="forestgreen",size=0.6)+  
#     geom_hline(yintercept=M_good,linetype=2,colour="blue3",size=0.6)+  
#     geom_hline(yintercept=M_bad,linetype=2,colour="red2",size=0.6)+  
#     
#     annotate("text", x = 0.5, y = 1.05, label=paste("KS=", round(M_KS, 4), "at Pop=", round(Pop, 4)), size=4, alpha=0.8)+  
#     
#     scale_x_continuous(breaks=seq(0,1,.2))+  
#     scale_y_continuous(breaks=seq(0,1,.2))+  
#     
#     xlab("of Total Population")+  
#     ylab("of Total Bad/Good")+  
#     
#     ggtitle(label="KS - Chart")+  
#     
#     theme_bw()+  
#     
#     theme(  
#       plot.title=element_text(colour="gray24",size=12,face="bold"),  
#       plot.background = element_rect(fill = "gray90"),  
#       axis.title=element_text(size=10),  
#       axis.text=element_text(colour="gray35")  
#     )  
#   
#   result<-list(M_KS=M_KS,Pop=Pop,PlotKS=PlotKS,df_ks=df_ks)  
#   return(result)  
# }  
# 
# '''测试集ks'''
# PlotKS_N(lgb.pred,lgb_te3$target,1,100)
# '''时间外样本ks'''
# PlotKS_N(lgb.pred.OTT,data_OTT$target,1,100)

