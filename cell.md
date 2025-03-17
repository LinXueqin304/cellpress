#testrt=read.table("../00.TEST数据/brca_metabric/data_mrna_illumina_microarray_zscores_ref_diploid_samples.txt", header=T, sep="\t")
#testrt<-testrt[-2]
#test <- aggregate( . ~ Hugo_Symbol,data=testrt, max)
#rownames(test)<-test$Hugo_Symbol
#test<-test[-1]
#test<-t(test)
#testclinic<-data.table::fread("../../00.TEST数据/brca_metabric/brca_metabric/data_clinical_patient.txt", sep="\t",skip = "PATIENT_ID")
#df_testclinic <- as.data.frame(testclinic)
#cleaned_df <- na.omit(df_testclinic)
#cleancli1=cleaned_df[apply(cleaned_df != "", 1, all), ]
#samples=cleancli1$PATIENT_ID
#testclinic<-data.table::fread("../../00.TEST数据/brca_metabric/KM_Plot__Overall.txt", sep="\t",header = T)
#testclinic<-na.omit(testclinic)
#tclinic<-cbind(testclinic$Patient.ID,testclinic$OS_MONTHS,testclinic$OS_STATUS)
#tclinic<-as.data.frame(tclinic)
#row.names(tclinic)<-tclinic$V1
#colnames(tclinic)<-c("id","futime","fustat")
#tclinic$fustat <- unlist(sapply(tclinic$fustat, function(x) strsplit(x,":")[[1]][1]))
#tclinic$futime<- as.numeric(tclinic$futime) /12
#row.names(tclinic)<-str_replace_all(row.names(tclinic), "-", ".")
#sameSample=intersect(row.names(tclinic),row.names(test))
#tdata=test[sameSample,]
#tcli=tclinic[sameSample,]
#texp=cbind(tcli,tdata)
#ids=str_replace_all(samples, "-", ".")
#texp1=texp[ids,]
#texp=texp1
#save(tdata,tcli,texp,file="test数据集.RData")
#library(dplyr)
#library("rjson")
#library(SummarizedExperiment)
#library(stringr)
#setwd("E:/工作/项目2-乳腺癌/")
#json <- jsonlite::fromJSON("00.TCGA数据/metadata.cart.2023-07-11.json")
#sample_id <- sapply(json$associated_entities,function(x){x[,1]})  
#file_sample <- data.frame(sample_id,file_name=json$file_name)  
#count_file <- list.files('./00.TCGA数据/gdc_download_20230711_084645.127238/',pattern = '*gene_counts.tsv$',recursive = TRUE,full.names = TRUE)
#count_file_name <- strsplit(count_file,split='/')  
#count_file_name <- sapply(count_file_name,function(x){x[6]})
#matrix = data.frame(matrix(nrow=60660,ncol=0)) 
#for (i in count_file){
  #data<- read.delim(i,fill = TRUE,header = FALSE,row.names = 1)
  #colnames(data)<-data[2,]
  #data <-data[-c(1:6),]
  #取出COUNT矩阵:3，TPM:6，fpkm-unstranded:7，fpkm-up-unstranded:8
  #data <- data[6]
  #f=unlist(strsplit(i,split='/'))
  #fn=f[length(f)]
  #colnames(data) <- file_sample$sample_id[which(file_sample$file_name==fn)]
  #print(fn)
  #matrix <- cbind(matrix,data)
#}
#将表达量矩阵的列名设置为Gene Symbol
#path = count_file[1]
#data<- as.matrix(read.delim(path,fill = TRUE,header = FALSE,row.names = 1))
#gene_name <-data[-c(1:6),1]
#matrix0 <- cbind(gene_name,matrix)
#将gene_name列去除重复的基因，保留每个基因最大表达量结果(非常非常慢)
#matrix0 <- aggregate( . ~ gene_name,data=matrix0, max)    
#去除数值全部为0的行
#matrix0=matrix0[rowMeans(matrix0)>0,]
#rownames(matrix0) <- matrix0[,1]
#exp <- matrix0[,-1]
# write.csv(exp,'exprSet.csv',row.names = TRUE)
#载入临床特征数据，并进行其中缺失数据的清洗
#clinical_data_original <- jsonlite::fromJSON('TCGA数据/clinical.cart.2023-07-11.json')
#clinical_data_original<-clinical_data_original %>% dplyr::filter(!is.na(demographic$submitter_id))
#days_to_last_follow_up <- as.numeric(sapply(clinical_data_original$diagnoses, function(x) x$days_to_last_follow_up))
#stage=sapply(clinical_data_original$diagnoses, function(x) ifelse(is.null(x$ajcc_pathologic_stage), NA, x$ajcc_pathologic_stage))
#T_stage=sapply(clinical_data_original$diagnoses, function(x) ifelse(is.null(x$ajcc_pathologic_t), NA, x$ajcc_pathologic_t))
#M_stage=sapply(clinical_data_original$diagnoses, function(x) ifelse(is.null(x$ajcc_pathologic_m), NA, x$ajcc_pathologic_m))
#N_stage=sapply(clinical_data_original$diagnoses, function(x) ifelse(is.null(x$ajcc_pathologic_n), NA, x$ajcc_pathologic_n))
#clinical_trait <- clinical_data_original$demographic[,c('submitter_id', 'gender', 'days_to_death', 'vital_status',"age_at_index")] %>% #cbind(.,days_to_last_follow_up,stage,T_stage,M_stage,N_stage)
#clinical_trait <- tidyr::separate(clinical_trait, col= 'submitter_id',
                                  #into = c('submitter_id', 'junk'), sep='_', remove = T)[,-2]
#clinical_trait <- clinical_trait[!duplicated(clinical_trait$submitter_id),]
#clinical_trait[is.na(clinical_trait$days_to_death), 'days_to_death'] <- clinical_trait[is.na(clinical_trait$days_to_death), 'days_to_last_follow_up']
#clinical_trait <- clinical_trait[,-6]
#colnames(clinical_trait) [1:5]<- c('submitter_id', 'gender', 'futime', 'fustat',"age")
#clinical_trait <- na.omit(clinical_trait)
#2 区分肿瘤和正常样本
#condition_table <- tidyr::separate(data=data.frame('ids'=colnames(expr_count)[2:length(expr_count)]),
                                   #col='ids',
                                   #sep='-',
                                   #into=c('c1','c2','c3','c4','c5','c6','c7'))
#condition_table <- cbind('ids'=colnames(expr_count)[2:length(expr_count)], condition_table) %>%
  #cbind('submitter_IDs'=substr(colnames(expr_count)[2:length(expr_count)], 1, 12))
#condition_table$c4 <- gsub(condition_table$c4, pattern = '^0..', replacement = 'cancer')
#condition_table$c4 <- gsub(condition_table$c4, pattern = '^1..', replacement = 'normal')
#condition_table <- condition_table[,c('ids','c4','submitter_IDs')]
#names(condition_table) <- c('TCGA_IDs', 'sample_type', 'submitter_id')
#condition_table$submitter_id <- as.character(condition_table$submitter_id)
#condition_table$TCGA_IDs <- as.character(condition_table$TCGA_IDs)
#condition_table_cancer <- condition_table[condition_table$sample_type=='cancer',]
#本研究只使用肿瘤样本，提取肿瘤样本的ID与临床数据合并
#condition_table_cancer_filter <- condition_table_cancer[!duplicated(condition_table_cancer$submitter_id),] %>%
  #dplyr::inner_join(clinical_trait, by='submitter_id')
#mygroup<-condition_table_cancer_filter[-2]
#mygroup<-mygroup[-2]
#3 统一临床样本与表达量矩阵样本
#mRNA_exprSet <- exp[,mygroup$TCGA_IDs]
#4 输出最终临床文件
#save(mygroup,mRNA_exprSet,file="初始数据.RData")
#write.csv(mygroup,'mygroup.csv',row.names = TRUE)
#write.csv(mRNA_exprSet,'mRNA_exprSet.csv',row.names = TRUE,col.names = T)
#整理干性基因集
#stem_gene_set<-read.xlsx2("26个干性基因集.xlsx",sheetIndex = 1,startRow = 2)
#stem_gene_set<-t.data.frame(stem_gene_set)
#write.table(stem_gene_set, file = "gs.gmt", sep = "\t", row.names = TRUE, col.names = FALSE, quote = FALSE)
#geneSets=getGmt("gs.gmt", geneIdType=SymbolIdentifier())
#输入表达矩阵
#exp<- as.matrix(mRNA_exprSet) 
#dimnames=list(rownames(exp),colnames(exp))
#将矩阵转化为数值型
#data=matrix(as.numeric(as.matrix(exp)),nrow=nrow(exp),dimnames=dimnames)
#进行gsva分析
#ssgseaScore <- gsva(data, geneSets, method="ssgsea",
                    mx.diff=FALSE, verbose=FALSE)  
#normalize=function(x){
  #return((x-min(x))/(max(x)-min(x)))}
#ssgseaScore=normalize(ssgseaScore)
#输出ssGSEA结果
#ssgseaOut=rbind(id=colnames(ssgseaScore), ssgseaScore)
#write.table(ssgseaOut,file="ssGSEA.result.txt",sep="\t",quote=F,col.names=F)
#dir.create("./cluster2/")
#title="./cluster2/"
#rt=read.table("ssGSEA.result.txt", header = TRUE, row.names = 1)
#rt<- as.matrix(rt) 
#dimnames=list(rownames(rt),colnames(rt))
#将矩阵转化为数值型
#data=matrix(as.numeric(as.matrix(rt)),nrow=nrow(rt),dimnames=dimnames)
#cluster_result <- ConsensusClusterPlus(data, maxK = 10, reps = 1000, pItem = 0.8, pFeature = 1, clusterAlg = "km", title = "cluster2",plot="png", writeTable = TRUE)
#icl = calcICL(cluster_result,title=title,plot="png")
##绘制热图
#dir.create("./ssGSEA/")
#setwd("ssGSEA/")
# rt<-ssGSEArt
# rt=read.table("../ssGSEA.result.txt", header=T, sep="\t", check.names=F, row.names=1)
# mygroup=read.csv('../clinical_trait.csv',row.names = 1)
#mygroup$TCGA_IDs<-str_replace_all(mygroup$TCGA_IDs, "-", ".")
#rt=rt[order(rowMeans(rt)),]
#rownames(mygroup)<-mygroup$TCGA_IDs
#mygroup<-mygroup[-1]
#mygroup<-mygroup[colnames(rt),]
###输入cluster信息c1/c2/c3
#mycluster=read.csv('02.cluster/cluster2.k=2.consensusClass.csv',col.names = c("sample","cluster"))
#替换cluster名
#mycluster$cluster<- recode(mycluster$cluster, 
                          # "1" = "C1",
                          # "2" = "C2",
                          # "3" = "C3",
                          # "4" = "C4")
#mygroup<-cbind(mygroup[mycluster$sample,],mycluster$cluster)
#colnames(mygroup)[10]<-"cluster"
#mygroup<-mygroup[order(mygroup$cluster),]
#rt=rt[,row.names(mygroup)]
#绘图
#pdf(file="01.ssGSEA/heatmap.pdf", width=8, height=5)
#pheatmap(rt, 
         #annotation=mygroup, 
         #color=colorRampPalette(c(rep("blue",3), "white", rep("red",3)))(50),
         #cluster_cols=F,
         #show_colnames=F,
         #scale="row",
         #fontsize = 6,
         #fontsize_row=6,
         #fontsize_col=6)
#dev.off()
#setwd("E:/工作/项目2-乳腺癌/GO_KEGG/")
#data <- read.table(file = './symbol.txt', header = T)
#genes=read.table(file = './WGCNA.txt', header = F)
#genes=genes$V1
#ids <- bitr(genes,'SYMBOL','ENTREZID','org.Hs.eg.db')   
#gene_diff <- ids$ENTREZID
#eKEGG <- enrichKEGG(gene_diff,
                    #pvalueCutoff = 0.05,
                    #qvalueCutoff = 0.05,
#)
#dir.create('KEGG')
#write.csv(eKEGG, file = 'KEGG/KEGG.result.csv')
#eKEGG <- setReadable(eKEGG, OrgDb = org.Hs.eg.db, keyType="ENTREZID")
#KEGGresult=as.data.frame(eKEGG)
#eGO <- enrichGO(gene_diff,
                #pvalueCutoff = 0.05,
                #qvalueCutoff = 0.05,
                #OrgDb=org.Hs.eg.db
#)
#dir.create('GO')
#write.csv(eGO, file = 'GO/GO.result.csv')
#eGO <- setReadable(eGO, OrgDb = org.Hs.eg.db, keyType="ENTREZID")
#GOresult=as.data.frame(eGO)
#barplot(eKEGG, showCategory=20)+    
  #scale_y_discrete(labels=function(x) str_wrap(x, width = 100))
#barplot(eGO, showCategory=20)+    
  #scale_y_discrete(labels=function(x) str_wrap(x, width = 100))
#KEGG_result <- eKEGG@result
#interesting_set = c("Neuroactive ligand-receptor interaction","Systemic lupus erythematosus", "Staphylococcus aureus infection")
#barplot(eKEGG, showCategory=interesting_set) +
  #scale_y_discrete(labels=function(x) str_wrap(x, width = 100))
#ggplot(eKEGG, showCategory = 20, 
       #aes(GeneRatio, fct_reorder(Description, GeneRatio))) + 
  #geom_segment(aes(xend=0, yend = Description)) +
  #geom_point(aes(color=p.adjust, size = Count)) +
  #scale_color_viridis_c(guide=guide_colorbar(reverse=TRUE)) +
  #scale_size_continuous(range=c(2, 10)) +
  #theme_minimal() + 
  #ylab(NULL) 
#library(patchwork)
#geneList = genes
#names(geneList) <- genes
#p1 = cnetplot(eKEGG, foldChange=geneList,showCategory = 3)
#p2 = cnetplot(eKEGG, foldChange=geneList,showCategory = 3, circular = T)
#p3 = cnetplot(eKEGG, foldChange=geneList,showCategory = 3, colorEdge = T, node_label="gene")
#p4 = cnetplot(eKEGG, foldChange=geneList,showCategory = 3, layout = "gem")
#(p1 + p2) / (p3 + p4)
#heatplot(eKEGG, foldChange=geneList)
#res1 <- enrichplot::pairwise_termsim(eKEGG)
#emapplot(res1, showCategory = 15, layout="kk", cex_category=1.5,min_edge = 0.6) 
#setwd("E:/工作/项目2-乳腺癌/")
#mygroup=read.table('clinical.txt',header=T, sep="\t", check.names=F, row.names=1)
#ssGSEArt=read.table("ssGSEA.result.txt", header=T, sep="\t", check.names=F, row.names=1)
#install.packages(c("survival", "survminer"))
#library("survival")
#library("survminer")
#mygroup$fustat<- recode(mygroup$fustat, 
                        #"Alive" = "0",
                        #"Dead" = "1")
#mygroup$cluster<- recode(mygroup$cluster, 
                         #"C1" = "1",
                         #"C2" = "2",
                         #"C3" = "3",
                         #"C4" = "4")
#surv_data=cbind(mygroup[,3:4],mygroup[,ncol(mygroup)])
#colnames(surv_data)[3]="cluster"
#surv_data$futime<- as.numeric(surv_data$futime) /365
#surv_data$fustat<- as.numeric(surv_data$fustat) 
#surv_data$cluster<- as.numeric(surv_data$cluster) 
#fit <- survfit(Surv(futime,fustat ) ~ cluster, data =surv_data)
#print(fit)
#summary(fit)
#summary(fit)$table
#fittable<-summary(fit)$table
#d <- data.frame(time = fit$time,
                #n.risk = fit$n.risk,
                #n.event = fit$n.event,
                #n.censor = fit$n.censor,
                #surv = fit$surv,
                #upper = fit$upper,
                #lower = fit$lower
#)
#head(d)
#p.value <- 1-pchisq(surv_diff$chisq,df=1)
#p.value <- signif(p.value,3)
#p.value <- format(p.value,scientific=T)
#ggsurvplot(fit,data=surv_data,
           #pval=paste0("p = ",p.value),
           #xlab="Time(years)",
           #break.time.by = 2,
           #legend.title="Cluster",
           #legend.labs=c("C1", "C2","C3","C4"),
           #conf.int = TRUE,
           #risk.table = TRUE, 
           #risk.table.col = "strata",
           #linetype = "strata",
           #surv.median.line = "hv",
           #ggtheme = theme_bw(),
           #risk.table.title="",
           #size = 1.3
#) 
#dir.create("03.surv")
#save(fit,file = "03.surv/surv.RData")
#clinical<-read.table("../clinical.txt", header=T, sep="\t", check.names=F, row.names=1)
#exp<-read.csv("../GeneSymbol_TPM_matrix.csv", row.names=1)
#hubgenes<-read.table("../05.WGCNA/hubGenes_MMblue.txt", header=F, sep="\t", check.names=F)
#HUBexp<-exp[hubgenes$V1,]
#HUBexp<-t(HUBexp)
#sameSample=intersect(row.names(clinical),row.names(HUBexp))
#data=HUBexp[sameSample,]
#cli=clinical[sameSample,]
#cliexp=cbind(cli, data)
#write.table(cliexp, file="clinic.bulegenes.txt", sep="\t", row.names=F, quote=F)
#cliexp<-read.table("clinic.bulegenes.txt", header=T, sep="\t", check.names=F, row.names=1)
#library(survival)      
#pFilter=0.05          
#cliexp$futime<-cliexp$futime/365 
#cliexp<-cliexp[-1]
#outTab=data.frame()
#AllTab=data.frame()
#sigGenes=c("futime","fustat")
#for(gene in colnames(cliexp[,8:ncol(cliexp)])){
  #cox=coxph(Surv(futime, fustat) ~ cliexp[,gene], data = cliexp)
  #coxSummary = summary(cox)
  #coxP=coxSummary$coefficients[,"Pr(>|z|)"]
  #AllTab=rbind(AllTab,
               #cbind(gene=gene,
                     #HR=coxSummary$conf.int[,"exp(coef)"],
                     #HR.95L=coxSummary$conf.int[,"lower .95"],
                     #HR.95H=coxSummary$conf.int[,"upper .95"],
                     #pvalue=coxP) )
  #if(coxP<pFilter){
    #sigGenes=c(sigGenes,gene)
    #outTab=rbind(outTab,
                 #cbind(gene=gene,
                       #HR=coxSummary$conf.int[,"exp(coef)"],
                       #HR.95L=coxSummary$conf.int[,"lower .95"],
                       #HR.95H=coxSummary$conf.int[,"upper .95"],
                       #pvalue=coxP) )
  #}
#}
#outf="blue_hubs"
#dir.create(outf)
#write.table(AllTab,file=paste0(outf,"/ALLuniCox.txt"),sep="\t",row.names=F,quote=F)
#write.table(outTab,file=paste0(outf,"/uniCox.txt"),sep="\t",row.names=F,quote=F)
#surSigExp=cliexp[,sigGenes]
#surSigExp=cbind(id=row.names(surSigExp),surSigExp)
#write.table(surSigExp,file=paste0(outf,"/uniSigExp.txt"),sep="\t",row.names=F,quote=F)
#save(AllTab,file=paste0(outf,"/uniCox.RData"))
#bioForest=function(coxFile=null, forestFile=null, forestCol=null){
  #rt <- read.table(coxFile,header=T,sep="\t",row.names=1,check.names=F)
  #gene <- rownames(rt)
  #hr <- sprintf("%.3f",rt$"HR")
  #hrLow  <- sprintf("%.3f",rt$"HR.95L")
  #hrHigh <- sprintf("%.3f",rt$"HR.95H")
  #Hazard.ratio <- paste0(hr,"(",hrLow,"-",hrHigh,")")
  #pVal <- ifelse(rt$pvalue<0.001, "<0.001", sprintf("%.3f", rt$pvalue))
  #pdf(file=forestFile, width=16, height=20)
  #n <- nrow(rt)
  #nRow <- n+1
  #ylim <- c(1,nRow)
  #layout(matrix(c(1,2),nc=2),width=c(3,2.5))
  #xlim = c(0,3)
  #par(mar=c(4,2.5,2,1))
  #plot(1,xlim=xlim,ylim=ylim,type="n",axes=F,xlab="",ylab="")
  #text.cex=0.8
  #text(0,n:1,gene,adj=0,cex=text.cex)
  #text(2.08-0.5*0.2,n:1,pVal,adj=1,cex=text.cex);text(2.08-0.5*0.2,n+1,'pvalue',cex=text.cex,font=2,adj=1)
  #text(3.12,n:1,Hazard.ratio,adj=1,cex=text.cex);text(3.12,n+1,'Hazard ratio',cex=text.cex,font=2,adj=1,)
  #par(mar=c(4,1,2,1),mgp=c(2,0.5,0))
  #xlim = c(0,max(as.numeric(hrLow),as.numeric(hrHigh)*1.5))
  #plot(1,xlim=xlim,ylim=ylim,type="n",axes=F,ylab="",xaxs="i",xlab="Hazard ratio")
  #arrows(as.numeric(hrLow),n:1,as.numeric(hrHigh),n:1,angle=90,code=3,length=0.05,col="darkblue",lwd=2.5)
  #abline(v=1,col="black",lty=2,lwd=2)
  #boxcolor = ifelse(as.numeric(hr) > 1, forestCol[1], forestCol[2])
  #points(as.numeric(hr), n:1, pch = 15, col = boxcolor, cex=1.6)
  #axis(1)
  #dev.off()
#}
#coxFile <- paste0(outf,"/uniCox.txt") 
#forestFile <- paste0(outf,"/sig.Forest.pdf")
#forestCol <- c("red", "green")
#bioForest(coxFile = coxFile, forestFile = forestFile, forestCol = forestCol)
#setwd("E:/工作/项目2-乳腺癌/06.预后模型/")    
#unicox=read.table("../04.unicox/blue_hubs/ALLuniCox.txt", header = TRUE,row.names = 1)
#filtergenes=unicox[unicox$pvalue<0.01,]
#filtergenes=filtergenes[order(filtergenes$pvalue),]
#allgenes=rownames(filtergenes)
#data <- read.table("../04.unicox/blue_hubs/uniSigExp.txt", header = TRUE,row.names = 1)
#data=cbind(data[1:2],data[,allgenes])
#load("test/test数据集2.RData")
#gene_combinations <- list()
#for (i in 5:15) {
  #gene_combinations[[i]] <- combn(imp_genes, i)
#}
#results <- list()
#bestAUC=0
#for (i in 5:length(gene_combinations)) {
  #for (j in 1:ncol(gene_combinations[[i]])) {
    #selected_genes <- gene_combinations[[i]][, j]
    #selected_data <- data[, c("futime", "fustat", selected_genes)]
    #rt=selected_data
    #rt=rt[rt$futime>0.08,]
    #x <- as.matrix(rt[,c(3:ncol(rt))])
    #y <- data.matrix(Surv(as.numeric(rt$futime),as.numeric(rt$fustat)))
    #set.seed(123)
    #fit <- glmnet(x, y, family = "cox",maxit = 1000)
    #cvfit=cv.glmnet(x, y, family="cox", maxit=1000)
    #coef=coef(fit, s = cvfit$lambda.min)
    #index=which(coef != 0)
    #actCoef=coef[index]
    #lassoGene=row.names(coef)[index]
    #if (length(lassoGene) == 1) {
      #next 
   #}
   #geneCoef=cbind(Gene=lassoGene,Coef=actCoef)
    #trainFinalGeneExp=rt[,lassoGene]
    #myFun=function(x){crossprod(as.numeric(x),actCoef)}
    #trainScore=apply(trainFinalGeneExp,1,myFun)
    #outCol=c("futime","fustat",lassoGene)
    #Risk=as.vector(ifelse(trainScore>median(trainScore),"high","low"))
    #outTab=cbind(rt[,outCol],riskScore=as.vector(trainScore),Risk)
    #testFinalGeneExp=texp1[,lassoGene]
    #testScore=apply(testFinalGeneExp,1,myFun)
    #outCol=c("futime","fustat",lassoGene)
    #Risk=as.vector(ifelse(testScore>median(testScore),"high","low"))
    #toutTab=cbind(texp1[,outCol],riskScore=as.vector(testScore),Risk)
    #diff=survdiff(Surv(as.numeric(futime),as.numeric(fustat)) ~ Risk,data = toutTab)
    #pValue=1-pchisq(diff$chisq,df=1)
    #fit2 <- survfit(Surv(as.numeric(futime),as.numeric(fustat)) ~ Risk, data = toutTab)
    #fittable<-as.data.frame(summary(fit2)$table)
    #if (pValue< 0.05 && fittable$rmean[1]<fittable$rmean[2] ) {
      #print("P value is less than 0.05. Further analysis can be performed.")
      #risk=toutTab[,c("futime", "fustat", "riskScore")]
      #ROC_rt=timeROC(T=risk$futime, delta=risk$fustat,
                     #marker=risk$riskScore, cause=1,
                     #weighting='aalen',
                     #times=c(1,2,3), ROC=TRUE)
      
      #if (mean(ROC_rt$AUC) > bestAUC) {
        #bestcoef=geneCoef
        #bestTrain=outTab
        #bestTest=toutTab
        #bestAUC=mean(ROC_rt$AUC)
        #bestROC=ROC_rt
        #bestfit=fit
        #bestcvfit=cvfit
        #pdf("lasso.lambda.pdf")
        #plot(fit, xvar="lambda", label=TRUE)
        #dev.off()
        #pdf("lasso.cvfit.pdf")
        #plot(cvfit)
        #abline(v=log(c(cvfit$lambda.min,cvfit$lambda.1se)), lty="dashed")
        #dev.off()
        #if (mean(ROC_rt$AUC) > 0.7) {
          #print("Roc >0.7!!!.")
          #write.table(cbind(id=rownames(outTab),outTab),file="risk.TCGAfinal.txt",sep="\t",quote=F,row.names=F)
          #write.table(cbind(id=rownames(toutTab),toutTab),file="risk.testfinal.txt",sep="\t",quote=F,row.names=F)
          #write.table(geneCoef,file="geneCoef.txt",sep="\t",quote=F,row.names=F)
        #}
      #}
    #}
  #}
#}

#write.table(cbind(id=rownames(bestTrain),bestTrain),file="risk.TCGAbest.txt",sep="\t",quote=F,row.names=F)
#write.table(cbind(id=rownames(bestTest),bestTest),file="risk.testbest.txt",sep="\t",quote=F,row.names=F)
#write.table(bestcoef,file="bestgeneCoef.txt",sep="\t",quote=F,row.names=F)
#bestTest2=bestTest[bestTest$futime<10,]
#write.table(cbind(id=rownames(bestTest2),bestTest2),file="risk.testbest2.txt",sep="\t",quote=F,row.names=F)
#bioSurvival=function(inputFile=null){
  #rt=read.table(inputFile, header=T, sep="\t", check.names=F)
  #diff=survdiff(Surv(futime, fustat) ~ Risk,data = rt)
  #pValue=diff$pvalue
  #pValue=1-pchisq(diff$chisq,df=1)
  #if(pValue<0.001){
    #pValue="p<0.001"
  #}else{
    #pValue=paste0("p=",sprintf("%.03f",pValue))
  #}
  #fit <- survfit(Surv(futime, fustat) ~ Risk, data = rt)
  #surPlot=ggsurvplot(fit, 
                     #data=rt,
                     #conf.int=T,
                     #pval=pValue,
                     #pval.size=6,
                     #legend.title="Risk",
                     #legend.labs=c("High risk", "Low risk"),
                     #xlab="Time(years)",
                     #ylab="Overall survival",
                     #break.time.by = 1,
                     #palette=c("red", "blue"),
                     #risk.table=TRUE,
                     #risk.table.title="",
                     #risk.table.height=.25)
  #print(surPlot)
#}
#bioSurvival(inputFile="risk.testbest.txt")
#bioSurvival(inputFile="risk.TCGAbest.txt")
#riskFile="risk.TCGAbest.txt"
#riskFile="risk.testbest.txt"  
#cliFile="../clinical.txt"      
#risk=read.table(riskFile, header=T, sep="\t", check.names=F, row.names=1)
#risk=risk[,c("futime", "fustat", "riskScore")]
#cli=read.table(cliFile, header=T, sep="\t", check.names=F, row.names=1)
#samSample=intersect(row.names(risk), row.names(cli))
#risk1=risk[samSample,,drop=F]
#cli=cli[samSample,,drop=F]
#rt1=cbind(risk1, cli)
#bioCol=rainbow(ncol(rt1)-1, s=0.9, v=0.9)
#ROC_rt=timeROC(T=risk$futime, delta=risk$fustat,
               #marker=risk$riskScore, cause=1,
               #weighting='marginal',
               #times=seq(1, 10, by = 0.1), ROC=TRUE)
#plot(ROC_rt,time=1,col=bioCol[1],title=FALSE,lwd=2)
#plot(ROC_rt,time=1.6,col=bioCol[2],add=TRUE,title=FALSE,lwd=2)
#plot(ROC_rt,time=2.7,col=bioCol[3],add=TRUE,title=FALSE,lwd=2)
#legend('bottomright',
       #c(paste0('AUC at 1 years: ',sprintf("%.03f",ROC_rt$AUC[1])),
         #paste0('AUC at 2 years: ',sprintf("%.03f",ROC_rt$AUC[2])),
         #paste0('AUC at 3 years: ',sprintf("%.03f",ROC_rt$AUC[3]))),
       #col=bioCol[1:3], lwd=2, bty = 'n')
#setwd("E:/工作/项目2-乳腺癌/TME")
#source("Cibersort.R")
#library("reshape2")
#library("limma") 
#library("ggplot2") 
#library(ggpubr)
#library(ggsci)
#results=CIBERSORT("LM22.txt", "symbol.txt", perm=1000, QN=FALSE)
#write.csv(results, "./CIBERSORT_Results.csv")
#results=read.table("cibersort_result.txt",header=T, sep="\t", check.names=F, row.names=1)
#resfilter<-results[results$`P-value` < 0.05, ]
#TME_data <- as.data.frame(resfilter[,1:22])
#rownames(TME_data) <- str_replace_all(rownames(TME_data), "-", ".")
#mygroup=read.table("../06.预后模型/test/risk.TCGAbest.txt", header=T, sep="\t", check.names=F, row.names=1)
#mygroup<-mygroup[row.names(TME_data),]
#group_list <- mygroup$Risk %>% 
  #factor(.,levels = c("high","low"))
#TME_data$group <- group_list
#TME_data$sample <- row.names(TME_data)
#TME_New = melt(TME_data)
#colnames(TME_New)=c("Group","Sample","Celltype","Composition")  
#TME_New <- na.omit(TME_New)
#save(TME_New,file="TME.RData")
#if(T){
  #mytheme <- theme(plot.title = element_text(size = 12,color="black",hjust = 0.5),
                   #axis.title = element_text(size = 12,color ="black"), 
                   #axis.text = element_text(size= 12,color = "black"),
                   #panel.grid.minor.y = element_blank(),
                   #panel.grid.minor.x = element_blank(),
                   #axis.text.x = element_text(angle = 45, hjust = 1 ),
                   #panel.grid=element_blank(),
                   #legend.position = "top",
                   #legend.text = element_text(size=12),
                   #legend.title= element_text(size= 12)
  #) }

#box_TME <- ggplot(TME_New, aes(x = Celltype, y = Composition))+ 
  #labs(y="Cell composition",x= NULL,title = "TME Cell composition")+  
  #geom_boxplot(aes(colour =Group),size=0.8,position=position_dodge(0.5),width=0.4,outlier.colour=NULL,outlier.alpha = 0.3)+ 
  #scale_fill_manual(values = c("#1CB4B8", "#EB7369","yellow"))+
  #theme_classic() + mytheme + 
  #stat_compare_means(aes(group =  Group),
                     #label = "p.signif",
                     #method = "wilcox.test",
                     #hide.ns = T)+ scale_color_nejm()
#box_TME;ggsave("./TCGA_BRCA_TMErisk.pdf",box_TME,height=20,width=30,unit="cm")

                     hide.ns = T)+ scale_color_nejm()
#box_TME;ggsave("./TCGA_BRCA_TMErisk.pdf",box_TME,height=20,width=30,unit="cm")
#pFilter=0.001               
#expFile="symbol.txt"         
#riskFile="E:/工作/项目2-乳腺癌/06.预后模型/test/risk.TCGAbest.txt"     
#setwd("E:/工作/项目2-乳腺癌/07.药物敏感性分析")     
#data(cgp2016ExprRma)
#data(PANCANCER_IC_Tue_Aug_9_15_28_57_2016)
#allDrugs=unique(drugData2016$Drug.name)
#rt = read.table(expFile, header=T, sep="\t", check.names=F,row.names = 1)
#rt=as.matrix(rt)
#rownames(rt)=rt[,1]
#exp=rt
#exp=rt[,2:ncol(rt)]
#dimnames=list(rownames(exp),colnames(exp))
#data=matrix(as.numeric(exp),nrow=nrow(exp),ncol = ncol(exp),dimnames=dimnames)
#data=avereps(data)
#data=data[rowMeans(data)>0.5,]
#riskRT=read.table(riskFile, header=T, sep="\t", check.names=F, row.names=1)
#riskRT$riskScore[riskRT$riskScore>quantile(riskRT$riskScore,0.99)]=quantile(riskRT$riskScore,0.99)
#riskRT=riskRT[-5]
#for(drug in allDrugs){
  #print(drug)
  #possibleError=tryCatch(
    #{senstivity=pRRopheticPredict(data, drug, selection=1, dataset = "cgp2016")},
    #error = function(e) {
      #print("An error occurred:")
      #print(e)
      #e
    #})
  
  #if(inherits(possibleError, "error")){next}
  #senstivity=senstivity[senstivity!="NaN"]
  #senstivity[senstivity>quantile(senstivity,0.99)]=quantile(senstivity,0.99)
  #sameSample=intersect(row.names(riskRT), names(senstivity))
  #risk=riskRT[sameSample, c("riskScore","Risk"),drop=F]
  #senstivity=senstivity[sameSample]
  #rt=cbind(risk, senstivity)
  #rt$Risk=factor(rt$Risk, levels=c("low", "high"))
