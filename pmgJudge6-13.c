#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <string.h> 
#include <math.h>
#define bool _Bool
#define MAX_FILES 1000
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


int* read_pgm(const char* filename);
int view_pgm(const char *filename);
void reset();
void judge(const char *filename);
int first();
float* layOpt(int layNo,float *input);
void write_log(const char *filename, int result, float *judgelay, int n);
float randn(float mean, float stddev) {
    float z0;
    do {
        float u1 = (rand() + 1.0f) / (RAND_MAX + 2.0f);
        float u2 = (rand() + 1.0f) / (RAND_MAX + 2.0f);
        z0 = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
        z0 = z0 * stddev + mean;
    } while (z0 == 0.0f);
    return z0;
}
void allmalloc(bool teacher);
void backprop(int t,int answer);
int save(const char *filename);
int import(const char *filename);
void learning(const char *folderpath);
int time;
int AllLay;
int size;
int costNo;

float ***synapse; //synapse[X層目][入力層][出力層]
float **bias;
float **lay; //lay[X層目][y段目]  //layは入力層の値を保持する
float judgelay[10];
float ****CostSy; //CostSy[保存番号][X層目][入力層][出力層]  //CostSyは誤差を保持する
float ***CostBi; //CostBi[保存番号][X層目][y段目]  //CostBiはバイアスの誤差を保持する

int main() {
    
    time=0;
    AllLay =4;
    size=28*28;
    costNo = 10; // 誤差を保存する回数
    int importflag=0;

    

    allmalloc(1);

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
    printf("モードを選択してください（read/view/judge/reset/import/exit）: ");
    scanf("%9s", mode);



    if (strcmp(mode, "read") == 0) {
        printf("PGMファイル名を入力してください: ");
        scanf("%255s", filename);
        int *pixels = read_pgm(filename);
        if (!pixels) return 1;

        for (int i = 0; i <size; i++) {
            printf("%d ", pixels[i]);
        }
        free(pixels);
        
    } else if (strcmp(mode, "view") == 0) {
        printf("PGMファイル名を入力してください: ");
        scanf("%255s", filename);
        view_pgm(filename);
        return 1;
    } else if(strcmp(mode, "reset") == 0){
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
        printf("PGMファイルを含むフォルダを入力してください: ");
        char folderpath[512];
        scanf("%511s",&folderpath);
        learning(folderpath);
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

int view_pgm(const char *filename) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        perror("ファイルオープン失敗");
        return 1;
    }

    char magic[3];
    int width, height, maxval;
    fscanf(fp, "%2s", magic);
    if (magic[0] != 'P' || magic[1] != '5') {
        printf("PGM(P5)形式ではありません\n");
        fclose(fp);
        return 1;
    }
    fscanf(fp, "%d %d", &width, &height);
    fscanf(fp, "%d", &maxval);
    fgetc(fp); // 改行を読み飛ばす

    unsigned char *data = malloc(width * height);
    fread(data, 1, width * height, fp);

    const char *levels = " .:-=+*#%@";
    int n_levels = 10;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            unsigned char pixel = data[y * width + x];
            char c = levels[pixel * (n_levels - 1) / 255];
            putchar(c);
        }
        putchar('\n');
    }

    free(data);
    fclose(fp);

    printf("エンターキーを押すと終了します...");
    getchar(); getchar(); // 2回呼ぶことでscanfの改行も消費
    return 0;
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
        
        if(l== 0) {
            float *tmp = malloc(sizeof(float) * size);
            for(int i = 0; i < size; i++) {
                tmp[i]=(float)pixels[i]/255.0f;

                lay[l][i] = tmp[i] ;  //lay0にpixelsの値を代入

            }
            free(tmp);
            
        } 


        if(l == AllLay-1) {
            
            float *tmp = layOpt(l,lay[l]);
            for(int i = 0; i < 10; i++) {
                judgelay[i] = tmp[i];
            }
          
            

        }else{

            float *tmp = layOpt(l,lay[l]);

            for(int i = 0; i < size; i++) {

                lay[l+1][i] = tmp[i];

            }
        }
    }



    



    

    int max=0;

    for(int i = 0; i < 10; i++) {
        printf("判定値[%d]: %f\n", i, judgelay[i]);


        if(judgelay[i] > judgelay[max]) {
            max = i;
        }
    }


    

    printf("判定結果: %d\n", max);
    printf("正解の数字ラベルを入力してください。\n");
    int answer;
    scanf("%d", &answer);
    // 損失関数


        float *loss = (float*)malloc(sizeof(float) * costNo);
        for (int i = 0; i < costNo; i++) {
            loss[i] = 0.0f;
        }
        for (int i = 0; i < 10; i++) {
            float y = (i == answer) ? 1.0f : 0.0f;
            // log(total[i]) は total[i] > 0 のときのみ有効
            if (judgelay[i] > 0.0f) {
                loss[time] -= y * logf(judgelay[i]);
            }
        }
        if(time == 9){
            float sum = 0.0f;
            for(int i=0;i<costNo;i++){
                sum += loss[i];
            }
            printf("損失関数%f\n", sum/(float)costNo);
        }
        free(loss);
        


    //write_log(filename, max, judgelay, 100);
    backprop(time,answer); // 誤差逆伝播法を実行

   

   
    free(pixels);



    return;
}

float* layOpt(int layNo,float *input){ 

    float *total = malloc(sizeof(float) * size);

    if(layNo!=AllLay-1){
        for(int i = 0; i < size; i++) {
            total[i] = 0.0f;

                for(int j = 0; j < size; j++) {
                    total[i] += input[j]*synapse[layNo][j][i];
                }
                total[i] += bias[layNo][i];
            
        }

        for(int i = 0; i < size; i++) {
            total[i] = fmaxf(0.0f, total[i]);

        }

        
            
    }else if(layNo==AllLay-1){

        for(int i = 0; i < 10; i++) {

            total[i] = 0.0f;

                for(int j = 0; j < size; j++) {
                    total[i] += input[j]*synapse[layNo][j][i];
                }

                total[i] += bias[layNo][i];
            printf("total[%d]=%.20f\n", i, total[i]);
        }
        float max_val = total[0];

        for(int i = 1; i < 10; i++) {
            if (total[i] > max_val) max_val = total[i];
        }
        float sum= 0.0f;

        for(int i = 0; i < 10; i++) {
            total[i] = expf(total[i] - max_val);
            sum += total[i];

        }

        if (sum == 0.0f) sum = 1e-8f; 
            for(int i = 0; i < 10; i++) {
                total[i] /= sum;
            }
    }


            
                



    return total;


    
}

void write_log(const char *filename, int result, float *judgelay, int n) {
    FILE *fp = fopen("judge_log.txt", "a");
    if (!fp) {
        printf("ログファイルを書き込めませんでした。\n");
        return;
    }
    fprintf(fp, "ファイル名: %s\n", filename);
    fprintf(fp, "判定結果: %d\n", result);
    fprintf(fp, "判定値: ");
    for (int i = 0; i < 10; i++) {
        fprintf(fp, "%f ", judgelay[i]);
    }
    fprintf(fp, "\n---------------------------\n");
/*
    for (int l = 0; l < AllLay; l++) {
        fprintf(fp, "\n[Layer %d]\n ", l);
        for (int i = 0; i < size; i++) {
            fprintf(fp, "%f ", lay[l][i]);
        }
        fprintf(fp, "\n");
    }
    
*/
    //各層のバイアス値を出力
    for (int l = 0; l < size; l++) {
        fprintf(fp, "[nueron%d] ", l);
        for (int i = 0; i < size; i++) {
            fprintf(fp, "%f ", synapse[2][l][i]);
        }
        fprintf(fp, "\n");
    }


    fprintf(fp, "\n--------------------------\n");
    fclose(fp);
}

void backprop(int t,int answer) {




    float d_cost;
    //出力層
    for(int i=0;i<10;i++){ //layの値を0に初期化
        float y = (i == answer) ? 1.0f : 0.0f;
        float d_cost = judgelay[i] - y; // 誤差を計算
        CostBi[t][AllLay-1][i] = d_cost; // バイアスの誤差を計算
        for(int j = 0; j < size; j++) {
            CostSy[t][AllLay-1][j][i]=lay[AllLay-2][j]*d_cost;
        }
      
    }
    //中間層
        for(int l = AllLay-2; l >= 1; l--) { //ｌ＝３層目→２層目→１層目

            for(int i = 0; i < size; i++) {
                float total = 0.0f;
                if(lay[l][i] > 0.0f){
                    for(int j = 0; j < size; j++) {
                    total += synapse[l][i][j] * CostBi[time][l+1][j];
                    }
                }
                
                CostBi[t][l][i] = total;
                for(int j = 0; j < size; j++) {
                CostSy[t][l][j][i] = lay[l-1][j] * total;
                }
            }
        }

    
        

        time++;
        printf("%d",time);


    if(time >= costNo) {

        for(int l=0; l < AllLay; l++) {
            for(int i=0;i<size;i++){ 
                for(int j =0;j<size;j++){
                    float totalSy = 0.0f;
                    float totalBi = 0.0f;    
                    for(int t = 0; t < costNo; t++) {
                        totalSy += CostSy[t][l][i][j]; //誤差の合計を計算
                        totalBi += CostBi[t][l][j]; //バイアスの誤差の合計を計算
                        
                    }
                    float learn_scale=0.0025f;
                    
                    synapse[l][i][j] -= totalSy *(float)(AllLay-l)*learn_scale / (float)time;
                    bias[l][j] -= totalBi *(float)(AllLay-1)*learn_scale/ (float)time; //バイアスの更新
                }
            }

        }

        time = 0; // 時間がcostNoを超えたらリセット
        printf("ニューロン及びバイアスを実行しました。\n");
    }

    return; // 関数の終了




}

void allmalloc(bool teacher) {


             // 動的メモリ割り当て
    lay = malloc(sizeof(float*) * AllLay);
    for (int i = 0; i < AllLay; i++) {
        lay[i] = malloc(sizeof(float) * size); 
    }

    bias =malloc(sizeof(float*) * AllLay);
    for(int i = 0; i < AllLay; i++) {
        bias[i] = malloc(sizeof(float) * size); 
    }

    synapse = malloc(sizeof(float*) * AllLay);


    for (int i = 0; i < AllLay; i++) {
        synapse[i] = malloc(sizeof(float*) * size); // 2次元目

        for (int j = 0; j < size; j++) {
            synapse[i][j] = malloc(sizeof(float) * size); // 3次元目
        }
    }
    

    if(teacher==1){

        CostSy = malloc(sizeof(float***) * costNo); // 1次元目
        for (int i = 0; i < costNo; i++) {
            CostSy[i] = malloc(sizeof(float**) * AllLay); // 2次元目
            for (int j = 0; j < AllLay; j++) {
                CostSy[i][j] = malloc(sizeof(float*) * size); // 3次元目
                for (int k = 0; k < size; k++) {
                    CostSy[i][j][k] = malloc(sizeof(float) * size); // 4次元目
                }
            }
        }


        CostBi = malloc(sizeof(float*) * costNo);
        for (int i = 0; i < costNo; i++) {
            CostBi[i] = malloc(sizeof(float*) * AllLay); // 2次元目

            for (int j = 0; j < AllLay; j++) {
                CostBi[i][j] = malloc(sizeof(float) * size); // 3次元目
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
                fprintf(fp, "%d %d %d %f\n", l, i, j, synapse[l][i][j]);
            }
        }
    }
    // biasの保存
    fprintf(fp, "# bias\n");
    for (int l = 0; l < AllLay; l++) {
        for (int i = 0; i < size; i++) {
            fprintf(fp, "%d %d %f\n", l, i, bias[l][i]);
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
        float val;
        if (sscanf(line, "%d %d %d %f", &l, &i, &j, &val) == 4) {
            synapse[l][i][j] = val;
        }
    }
    // biasの読み込み
    do {
        int l, i;
        float val;
        if (sscanf(line, "%d %d %f", &l, &i, &val) == 3) {
            bias[l][i] = val;
        }
    } while (fgets(line, sizeof(line), fp));

    fclose(fp);
    printf("パラメータを%sから読み込みました。\n", filename);
    return 1;
}

void learning(const char *folderpath) {
    DIR *dir;
    struct dirent *ent;
    char filepath[512];
    int count = 0;

    dir = opendir(folderpath);
    if (dir == NULL) {
        printf("フォルダが開けません: %s\n", folderpath);
        return;
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
    printf("フォルダ内の全ファイルをjudgeしました。\n");
}
