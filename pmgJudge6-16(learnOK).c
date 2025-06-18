
#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <string.h> 
#include <math.h>
#define bool _Bool
#define MAX_FILES 100000
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define DROPOUT_RATE 0.33// 50%の確率でノードを無効化


int* read_pgm(const char* filename);
int view_pgm(const char *filename);
void reset();
void judge(const char *filename);
int first(int importflag);
double* layOpt(int layNo,double *input);
void write_log(const char *filename, int result, double *judgelay, int n);
double randn(double mean, double stddev) {
    double z0;
    do {
        double u1 = (rand() + 1.0f) / (RAND_MAX + 2.0f);
        double u2 = (rand() + 1.0f) / (RAND_MAX + 2.0f);
        z0 = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
        z0 = z0 * stddev + mean;
    } while (z0 == 0.0f);
    return z0;
}
void allmalloc(bool teacher);
void backprop(int t,int answer);
int save(const char *filename);
int import(const char *filename);
int learning(const char *folderpath);
void init_dropout_mask();
int time;
int correct;
int AllLay;
int size;
int batchNum;
float totalCost;
float corectRate;
int** dropout_mask; // [層][ノード]  各ノードのマスク（0または1）

double ***synapse; //synapse[X層目][入力層][出力層]
double **bias;
double **lay; //lay[X層目][y段目]  //layは入力層の値を保持する
double judgelay[10];

double **totalBias;//totalBI[X層目][y段目]  //誤差を保持する用
double ***totalSynapse;//totalSy[X層目][入力層][出力層]  //誤差を保持する用

int main() {
    
    time=0;
    AllLay =4;
    size=28*28;
    batchNum = 200; // 誤差を保存する回数 
    int importflag=0;

    

    allmalloc(1);
    init_dropout_mask();

   int temp = 1;

    while(temp==1){
        temp = first(importflag);
        importflag=1;
    }
    
    while(1){
        printf("セーブしますか(save/exit)\n");
        char mode[10];
        scanf("%9s", mode);
        if (strcmp(mode, "save") == 0){
            save("savedate.param");
            break;
        }else if(strcmp(mode, "exit") == 0){
            break;
        }

    }


    //メモリの解放

    return 0;
}

int first(int importflag){


        char mode[10];

        while(importflag==0){
            printf("importもしくはresetを選択してください。(import/reset)");
            scanf("%9s", mode);



            if(strcmp(mode, "reset") == 0){
                reset();
                printf("ニューラルネットワークを初期化しました。\n");
                importflag=1;
            }else if(strcmp(mode, "import") == 0){
                int i;
                i =import("savedate.param");
                if (i==0){
                    char importfile[256] ;
                    printf("savadate.paramを入力してください: ");
                    scanf("%255s", importfile);
                    import(importfile);
                    importflag=1;

                }else{
                    importflag=1;
                }
            }
        }
        

    
    
    char filename[256];
    printf("モードを選択してください（read/judge/reset/import/exit）: ");
    scanf("%9s", mode);



    if(strcmp(mode, "reset") == 0){
        reset();
        printf("ニューラルネットワークを初期化しました。\n");
        return 1;
    } else if(strcmp(mode, "judge") == 0) {
        char filename[256];
        printf("PGMファイル名を入力してください: ");
        scanf("%255s", filename);
        judge(filename);
        return 1;
    }else if (strcmp(mode, "learn") == 0){
        int count =0;
        printf("PGMファイルを含むフォルダを入力してください: ");
        char folderpath[512];
        scanf("%511s",&folderpath);
        count=learning(folderpath);

        printf("エポックの損失関数%lf\n", totalCost/(count/batchNum));
        printf("エポックの正答率%f%%\n", corectRate/(count/batchNum));

        totalCost=0;
        corectRate=0;



        return 1;

    } else if(strcmp(mode, "import") == 0){
        int i;
        i =import("savedate.param");
        if (i==0){
            char importfile[256] ;
            printf("savadate.paramを入力してください: ");
            scanf("%255s", importfile);
            import(importfile);

        }
        return 1;
    } else if(strcmp(mode, "exit") == 0) {
        return 0;
    } else {
        printf("不正なモードです。\n");
        return 1;
    }

}

int* read_pgm(const char* filename) {

    int width;
    int height;

    FILE* fp = fopen(filename, "rb");//fpには読み込み開始地点が記録される
    if (!fp) {
        printf("ファイルを開けません: %s\n", filename);
        return NULL;
    }

    char magic[3];//先頭から３文字取得
    fscanf(fp, "%2s", magic);


    int c;
    while ((c = fgetc(fp)) == '#') {
        while (fgetc(fp) != '\n'); // その行をスキップ
    }
        //1文字余分で読み込んだので、ungetcで戻す
    ungetc(c, fp);
        
    int maxval;

    fscanf(fp, "%d %d %d", &width, &height,&maxval);

    fgetc(fp); // 改行読み飛ばし


    int* data = (int*)malloc(sizeof(int) * size);

    if (strcmp(magic, "P5") == 0) {
        // バイナリ形式
        for (int i = 0; i < size; i++) {
            data[i] = fgetc(fp);
        }
    } else {
        printf("未対応のPGM形式です: %s\n", magic);
        free(data);
        fclose(fp);
        return NULL;
    }

    fclose(fp);
    return data;
}

void reset() {
    
    for(int l = 0; l < AllLay; l++) {
        for(int i = 0; i < size; i++) {
            for(int j = 0; j < size; j++) {
                synapse[l][i][j]=randn(0.0f, 0.01f); //He初期化
            }
            
            bias[l][i]=0;
        }
        
    }

}

void judge(const char *filename) {

    int *pixels = read_pgm(filename);
 
    if (!pixels) return ;

    for(int l = 0; l < AllLay; l++) {
        double *tmp = malloc(sizeof(double) * size);
        if(l== 0) {                             //初めにlay[0]にpgmデータを入れる
            for(int i = 0; i < size; i++) {
                lay[l][i]  =(double)pixels[i]/255.0f;

            }
        }


        if(l == AllLay-1) {
            
            double *tmp = layOpt(l,lay[l]);
            for(int i = 0; i < 10; i++) {
                judgelay[i] = tmp[i];
            }
            

        }else{
            double *tmp = layOpt(l,lay[l]);
            for(int i = 0; i < size; i++) {
                lay[l+1][i] = tmp[i];
            }
        }
        
        free(tmp);
    }

    
    int max=0;

    for(int i = 0; i < 10; i++) {
        printf("判定値[%d]: %lf\n", i, judgelay[i]);


        if(judgelay[i] > judgelay[max]) {
            max = i;
        }
    }

    

    printf("判定結果: %d\n", max);
    printf("正解の数字ラベルを入力してください。\n");
    int answer;
    
    scanf("%d", &answer);

    if(max==answer){
        correct +=1;
    }
    // 損失関数


        double *loss = (double*)malloc(sizeof(double) * batchNum);
        for (int i = 0; i < batchNum; i++) {
            loss[i] = 0.0f;
        }
        for (int i = 0; i < 10; i++) {
            double y = (i == answer) ? 1.0f : 0.0f;
            // log(total[i]) は total[i] > 0 のときのみ有効
            if (judgelay[i] > 0.0f) {
                loss[time] -= y * logf(judgelay[i]);
            } 

        }

        if(time == batchNum-1){
            double sum = 0.0f;
            for(int i=0;i<batchNum;i++){
                sum += loss[i];
            }
            totalCost += sum/(double)batchNum;
            corectRate += (float)correct/(float)batchNum*100;
            
            printf("損失関数%lf\n", sum/(double)batchNum);
            printf("正答率%f%%\n",(float)correct/(float)batchNum*100);
            correct=0;
            printf("シナプスとバイアスを更新しています...");
            //write_log(filename, max, judgelay, 100);
        }

        backprop(time,answer); // 誤差逆伝播法を実行
        
            
    free(pixels);



    return;
}

double* layOpt(int l,double *input){ 

    double *total = malloc(sizeof(double) * size);

    if(l!=AllLay-1){
        for(int i = 0; i < size; i++) {
            total[i] = 0.0f;

            for(int j = 0; j < size; j++) {
                total[i] += input[j]*synapse[l][j][i];
            }
            total[i] += bias[l][i];
            

            total[i] = fmaxf(0.0f, total[i]);//ReLUに通す
            // ドロップアウト適用（学習時のみ）
            
            total[i] *= (float)dropout_mask[l][i];
   
            

        }

        
            
    }else if(l==AllLay-1){

        for(int i = 0; i < 10; i++) {

            total[i] = 0.0f;

                for(int j = 0; j < size; j++) {
                    total[i] += input[j]*synapse[l][j][i];
                }

                total[i] += bias[l][i];

                //printf("total[%d]:%f\n",i,total[i]);  //debug
        }

        double max_val = total[0];
        for(int i = 1; i < 10; i++) {
            if (total[i] > max_val) max_val = total[i];
        }

        double sum= 0.0f;
        for(int i = 0; i < 10; i++) {
            total[i] = expf(total[i] - max_val);
            sum += total[i];

        }

        if (sum == 0.0f) sum = 1e-20f; 
            for(int i = 0; i < 10; i++) {
                total[i] /= sum;
                
            }
            
    }



    return total;


    
}

void write_log(const char *filename, int result, double *judgelay, int n) {
    FILE *fp = fopen("judge_log.txt", "a");
    if (!fp) {
        printf("ログファイルを書き込めませんでした。\n");
        return;
    }
    fprintf(fp, "ファイル名: %s\n", filename);
    fprintf(fp, "判定結果: %d\n", result);
    fprintf(fp, "判定値: ");
    for (int i = 0; i < 10; i++) {
        fprintf(fp, "%lf ", judgelay[i]);
    }
    fprintf(fp, "\n---------------------------\n");
/*
    for (int l = 0; l < AllLay; l++) {
        fprintf(fp, "\n[Layer %d]\n ", l);
        for (int i = 0; i < size; i++) {
            fprintf(fp, "%lf ", lay[l][i]);
        }
        fprintf(fp, "\n");
    }
    
*/
    //各層のバイアス値を出力
    for (int l = 0; l < size; l++) {
        fprintf(fp, "[nueron%d] ", l);
        for (int i = 0; i < size; i++) {
            fprintf(fp, "%lf ", synapse[2][l][i]);
        }
        fprintf(fp, "\n");
    }


    fprintf(fp, "\n--------------------------\n");
    fclose(fp);
}

void backprop(int t,int answer) {

    double learn_scale=0.01;//学習率


    //出力層judge
   

    float delta[784];
    float total[784];
    for(int i=0;i<784;i++){
        total[i]=0;
    }

    for(int l = 3; l >= 0; l--){
        //bias[3] synapse[3]の更新
        if(l==AllLay-1){
            #pragma omp parallel for default(none) shared(delta,judgelay, answer, totalBias, totalSynapse, lay, total, size, l)
            for(int i=0;i<10;i++){ //layの値を0に初期化 //iは出力元
                double y = (i == answer) ? 1.0 : 0.0;
                double sum =0;

                sum = -y/judgelay[i] +(1-y)/(1-judgelay[i]); // 誤差を計算
                totalBias[l][i] += sum; 

                for(int j = 0; j < size; j++) {//jは入力元
                    totalSynapse[l][j][i] +=lay[l][j]*sum;
                }
                delta[i] = sum;

            }

        }else if(l==AllLay-2){
        //bias[2] synapse[2]の更新
            #pragma omp parallel for default(none) shared(synapse, delta, totalBias, totalSynapse, lay, total, size, l)
            for(int i = 0; i < size; i++) {//iは出力元  
                double sum =0;
                if(lay[l][i] > 0.0f){
                    for(int j = 0; j < 10; j++) {
                        sum += synapse[l+1][i][j]*delta[j];
                    }

                    totalBias[l][i] += sum;

                    for(int j = 0; j < size; j++) {//jは入力元
                    
                        totalSynapse[l][j][i] += lay[l][j] * sum;
                    }
                }
                
                total[i] =sum;

            }

            for(int i = 0; i < size; i++) {
                delta[i] = total[i];
                total[i]=0;
            }

    

        }else{
        #pragma omp parallel for default(none) shared(synapse, delta, totalBias, totalSynapse, lay, total, size, l)
        //bias[0-1] synapse[0-1]の更新           
            for(int i = 0; i < size; i++) {

                    double sum=0;
                    if(lay[l][i] > 0.0f){
                        for(int j = 0; j < size; j++) {
                            sum += synapse[l+1][i][j]*delta[j];
                        }
            
                            
                        totalBias[l][i] += sum;

                        for(int j = 0; j < size; j++) {
                            
                            totalSynapse[l][j][i] += lay[l][j] * sum;
                        }
                    }
                    total[i]=sum;
                
            }
            for(int i = 0; i < size; i++) {
                delta[i] = total[i];
                total[i]=0;
            }
        
        }

    }
    for (int l = 3; l >= 0; l--) {
        //bias[3] synapse[3]の更新
        if (l == AllLay - 1) {
    #pragma omp parallel for default(none) shared(delta,judgelay, answer, totalBias, totalSynapse, lay, total, size, l)
            for (int i = 0; i < 10; i++) { //layの値を0に初期化 //iは出力元
                double y = (i == answer) ? 1.0 : 0.0;
                double sum = 0;

                sum = -y / judgelay[i] + (1 - y) / (1 - judgelay[i]); // 誤差を計算
                totalBias[l][i] += sum;

                for (int j = 0; j < size; j++) {//jは入力元
                    totalSynapse[l][j][i] += lay[l][j] * sum;
                }
                delta[i] = sum;

            }

        }
        else if (l == AllLay - 2) {
            //bias[2] synapse[2]の更新
    #pragma omp parallel for default(none) shared(synapse, delta, totalBias, totalSynapse, lay, total, size, l)
            for (int i = 0; i < size; i++) {//iは出力元  
                double sum = 0;
                if (lay[l][i] > 0.0f) {
                    for (int j = 0; j < 10; j++) {
                        sum += synapse[l + 1][i][j] * delta[j];
                    }

                    totalBias[l][i] += sum;

                    for (int j = 0; j < size; j++) {//jは入力元

                        totalSynapse[l][j][i] += lay[l][j] * sum;
                    }
                }

                total[i] = sum;

            }

            for (int i = 0; i < size; i++) {
                delta[i] = total[i];
                total[i] = 0;
            }



        }
        else {
    #pragma omp parallel for default(none) shared(synapse, delta, totalBias, totalSynapse, lay, total, size, l)
            //bias[0-1] synapse[0-1]の更新           
            for (int i = 0; i < size; i++) {

                double sum = 0;
                if (lay[l][i] > 0.0f) {
                    for (int j = 0; j < size; j++) {
                        sum += synapse[l + 1][i][j] * delta[j];
                    }


                    totalBias[l][i] += sum;

                    for (int j = 0; j < size; j++) {

                        totalSynapse[l][j][i] += lay[l][j] * sum;
                    }
                }
                total[i] = sum;

            }
            for (int i = 0; i < size; i++) {
                delta[i] = total[i];
                total[i] = 0;
            }

        }

    }





        

            
        time++;
        printf("%d",time);

    if(time >= batchNum) {

        


        for(int l=0; l < AllLay; l++) {
            int node_num = (l == AllLay-1) ? 10 : size;
            for(int i=0;i<node_num;i++){ 
                
                
                
                bias[l][i] -= totalBias[l][i] * learn_scale / (double)batchNum; // バイアスの更新
                totalBias[l][i] = 0.0f;


                for(int j =0;j<size;j++){
                    
                    
                    synapse[l][j][i] -= totalSynapse[l][j][i] * learn_scale / (double)batchNum; //シナプスの更新
                    totalSynapse[l][j][i] = 0.0f;

                }
            }
        }
        for(int i=0;i<size;i++){
            for(int j=10;j<size;j++){
                
                synapse[3][i][j]=0.0;
            }
        }

        init_dropout_mask();
        time = 0; // 時間がbatchNumを超えたらリセット
        printf("ニューロン及びバイアスを実行しました。\n");
    }

    return; // 関数の終了




}

void allmalloc(bool teacher) {


             // 動的メモリ割り当て
    lay = malloc(sizeof(double*) * AllLay);
    for (int i = 0; i < AllLay; i++) {
        lay[i] = malloc(sizeof(double) * size); 
    }

    bias =malloc(sizeof(double*) * AllLay);
    for(int i = 0; i < AllLay; i++) {
        bias[i] = malloc(sizeof(double) * size); 
    }

    synapse = malloc(sizeof(double*) * AllLay);


    for (int i = 0; i < AllLay; i++) {
        synapse[i] = malloc(sizeof(double*) * size); // 2次元目

        for (int j = 0; j < size; j++) {
            synapse[i][j] = malloc(sizeof(double) * size); // 3次元目
        }
    }
    

    if(teacher==1){
        dropout_mask = malloc(sizeof(int*) * AllLay);
        for (int l = 0; l < AllLay; l++) {
            dropout_mask[l] = malloc(sizeof(int) * size);
        }

        totalBias =malloc(sizeof(double*) * AllLay);
        for(int i = 0; i < AllLay; i++) {
            totalBias[i] = malloc(sizeof(double) * size); 
        }

        totalSynapse = malloc(sizeof(double*) * AllLay);


        for (int i = 0; i < AllLay; i++) {
            totalSynapse[i] = malloc(sizeof(double*) * size); // 2次元目

            for (int j = 0; j < size; j++) {
                totalSynapse[i][j] = malloc(sizeof(double) * size); // 3次元目
            }
    }
    

    }
    return; // 関数の終了
}
 
int save(const char *filename){
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        printf("ファイルを開けません: %s\n", filename);
        return 0;
    }


    fprintf(fp, "# synapse\n");
    for (int l = 0; l < AllLay; l++) {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                if(synapse[l][i][j] != 0){
                    fprintf(fp, "%d %d %d %.20f\n", l, i, j, synapse[l][i][j]);
                }else{fprintf(fp, "%d %d %d 0.0\n", l, i, j, synapse[l][i][j]);
                }
            }
        }
    }
    // biasの保存
    fprintf(fp, "# bias\n");
    for (int l = 0; l < AllLay; l++) {
        for (int i = 0; i < size; i++) {
            if(bias[l][i] != 0){
                fprintf(fp, "%d %d %.20f\n", l, i, bias[l][i]);
            }else{
                fprintf(fp, "%d %d 0.0\n", l, i, bias[l][i]);
            }

            
        }
    }

    fclose(fp);
    
    printf("ネットワークパラメータを%sに保存しました。\n",filename);
    return 1;

}

int import(const char *filename) {

    FILE *fp = fopen(filename, "r");
    if (!fp) {
        printf("ファイルを開けません: %s\n", filename);
        return 0;
    }

    char line[256];
    // synapseの読み込み
    while (fgets(line, sizeof(line), fp)) {

        if (strncmp(line, "# synapse", 9) == 0) {
            break;
        }
    }
    while (fgets(line, sizeof(line), fp)) {
        if (strncmp(line, "# bias", 6) == 0) {
            break;
        }
        int l, i, j;
        double val;
        if (sscanf(line, "%d %d %d %lf", &l, &i, &j, &val) == 4) {
            synapse[l][i][j] = val;
        }
    }
    // biasの読み込み
    do {
        int l, i;
        double val;
        if (sscanf(line, "%d %d %lf", &l, &i, &val) == 3) {
            bias[l][i] = val;
        }
    } while (fgets(line, sizeof(line), fp));

    fclose(fp);
    printf("パラメータを%sから読み込みました。\n", filename);
    return 1;
}

int learning(const char *folderpath) {
    DIR *dir;
    struct dirent *ent;
    char filepath[512];
    int count = 0;

    dir = opendir(folderpath);
    if (dir == NULL) {
        printf("フォルダが開けません: %s\n", folderpath);
        return 0;
    }

    // ファイル名を昇順で格納
    char **filelist = malloc(sizeof(char*) * MAX_FILES);
    for (int i = 0; i < MAX_FILES; i++) {
    filelist[i] = malloc(256);
}




    while ((ent = readdir(dir)) != NULL && count < MAX_FILES) {
        // 拡張子が.pgmのファイルのみ
        if (strstr(ent->d_name, ".pgm")) {
            strcpy(filelist[count], ent->d_name);
            count++;
        }
    }
    closedir(dir);

    // ソート（昇順）
    for (int i = 0; i < count - 1; i++) {
        for (int j = i + 1; j < count; j++) {
            if (strcmp(filelist[i], filelist[j]) > 0) {
                char tmp[256];
                strcpy(tmp, filelist[i]);
                strcpy(filelist[i], filelist[j]);
                strcpy(filelist[j], tmp);
            }
        }
    }

    // 上から順にjudge
    for (int i = 0; i < count; i++) {
        snprintf(filepath, sizeof(filepath), "%s\\%s", folderpath, filelist[i]);
        printf("judge: %s\n", filepath);
        judge(filepath);
    }

    for (int i = 0; i < count; i++) {
        free(filelist[i]);
        }
        free(filelist);
    printf("フォルダ内の全%d個のpgmファイルをjudgeしました。\n", count);
    return count;
}

void init_dropout_mask() {
    for (int l = 0; l < AllLay - 1; l++) { // 出力層には通常ドロップアウトしない
        for (int i = 0; i < size; i++) {
            dropout_mask[l][i] = ((double)rand() / RAND_MAX < DROPOUT_RATE) ? 0 : 1;
        }
    }
}
