#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <dirent.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// 定数定義
#define MAX_FILES 100000
#define IMAGE_SIZE 784  // 28*28
#define OUTPUT_SIZE 10
#define LAYER_COUNT 4
#define DROPOUT_RATE 0.5
#define MAX_FILENAME 256
#define MAX_PATH 512

// エラーコード
#define SUCCESS 0
#define ERROR_FILE_NOT_FOUND -1
#define ERROR_MEMORY_ALLOCATION -2
#define ERROR_INVALID_FORMAT -3

// ニューラルネットワーク構造体
typedef struct {
    int layer_count;
    int image_size;
    int output_size;
    int batch_size;
    float learning_rate;
    
    // ネットワークパラメータ
    double ***synapse;     // [layer][input][output]
    double **bias;         // [layer][node]
    double **layer_output; // [layer][node]
    
    // 学習用
    double **total_bias;
    double ***total_synapse;
    
    // 統計情報
    float total_cost;
    float correct_rate;
    int time_step;
    int correct_count;
} NeuralNetwork;

// 関数プロトタイプ宣言
int init_neural_network(NeuralNetwork *nn, int batch_size, float learning_rate);
void free_neural_network(NeuralNetwork *nn);
double randn(double mean, double stddev);
int* read_pgm(const char* filename);
double* forward_pass(NeuralNetwork *nn, double *input, int is_training);
double* softmax(double *input, int size);
void backward_pass(NeuralNetwork *nn, int target, double *output);
int judge_image(NeuralNetwork *nn, const char *filename);
int train_network(NeuralNetwork *nn, const char *folder_path, const char *label_file);
void reset_network(NeuralNetwork *nn);
int save_network(const NeuralNetwork *nn, const char *filename);
int load_network(NeuralNetwork *nn, const char *filename);

// He初期化による正規分布ランダム値生成
double randn(double mean, double stddev) {
    static int has_spare = 0;
    static double spare;
    
    if (has_spare) {
        has_spare = 0;
        return spare * stddev + mean;
    }
    
    has_spare = 1;
    double u1 = (rand() + 1.0) / (RAND_MAX + 2.0);
    double u2 = (rand() + 1.0) / (RAND_MAX + 2.0);
    double z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    spare = sqrt(-2.0 * log(u1)) * sin(2.0 * M_PI * u2);
    
    return z0 * stddev + mean;
}

// ニューラルネットワーク初期化
int init_neural_network(NeuralNetwork *nn, int batch_size, float learning_rate) {
    nn->layer_count = LAYER_COUNT;
    nn->image_size = IMAGE_SIZE;
    nn->output_size = OUTPUT_SIZE;
    nn->batch_size = batch_size;
    nn->learning_rate = learning_rate;
    nn->total_cost = 0.0f;
    nn->correct_rate = 0.0f;
    nn->time_step = 0;
    nn->correct_count = 0;
    
    // メモリ確保
    nn->synapse = (double***)malloc(LAYER_COUNT * sizeof(double**));
    nn->bias = (double**)malloc(LAYER_COUNT * sizeof(double*));
    nn->layer_output = (double**)malloc(LAYER_COUNT * sizeof(double*));
    nn->total_bias = (double**)malloc(LAYER_COUNT * sizeof(double*));
    nn->total_synapse = (double***)malloc(LAYER_COUNT * sizeof(double**));
    
    if (!nn->synapse || !nn->bias || !nn->layer_output || 
        !nn->total_bias || !nn->total_synapse) {
        return ERROR_MEMORY_ALLOCATION;
    }
    
    // 各層のメモリ確保と初期化
    for (int l = 0; l < LAYER_COUNT; l++) {
        int input_size = (l == 0) ? IMAGE_SIZE : IMAGE_SIZE;
        int output_size = (l == LAYER_COUNT - 1) ? OUTPUT_SIZE : IMAGE_SIZE;
        
        nn->synapse[l] = (double**)malloc(input_size * sizeof(double*));
        nn->total_synapse[l] = (double**)malloc(input_size * sizeof(double*));
        nn->bias[l] = (double*)calloc(output_size, sizeof(double));
        nn->total_bias[l] = (double*)calloc(output_size, sizeof(double));
        nn->layer_output[l] = (double*)malloc(output_size * sizeof(double));
        
        if (!nn->synapse[l] || !nn->total_synapse[l] || !nn->bias[l] || 
            !nn->total_bias[l] || !nn->layer_output[l]) {
            return ERROR_MEMORY_ALLOCATION;
        }
        
        for (int i = 0; i < input_size; i++) {
            nn->synapse[l][i] = (double*)malloc(output_size * sizeof(double));
            nn->total_synapse[l][i] = (double*)calloc(output_size, sizeof(double));
            
            if (!nn->synapse[l][i] || !nn->total_synapse[l][i]) {
                return ERROR_MEMORY_ALLOCATION;
            }
            
            // He初期化
            for (int j = 0; j < output_size; j++) {
                nn->synapse[l][i][j] = randn(0.0, sqrt(2.0 / input_size));
            }
        }
    }
    
    return SUCCESS;
}

// メモリ解放
void free_neural_network(NeuralNetwork *nn) {
    if (!nn) return;
    
    for (int l = 0; l < nn->layer_count; l++) {
        int input_size = (l == 0) ? nn->image_size : nn->image_size;
        
        if (nn->synapse && nn->synapse[l]) {
            for (int i = 0; i < input_size; i++) {
                free(nn->synapse[l][i]);
            }
            free(nn->synapse[l]);
        }
        
        if (nn->total_synapse && nn->total_synapse[l]) {
            for (int i = 0; i < input_size; i++) {
                free(nn->total_synapse[l][i]);
            }
            free(nn->total_synapse[l]);
        }
        
        free(nn->bias[l]);
        free(nn->total_bias[l]);
        free(nn->layer_output[l]);
    }
    
    free(nn->synapse);
    free(nn->bias);
    free(nn->layer_output);
    free(nn->total_bias);
    free(nn->total_synapse);
}

// PGMファイル読み込み
int* read_pgm(const char* filename) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        printf("ファイルを開けません: %s\n", filename);
        return NULL;
    }
    
    char magic[3];
    fscanf(fp, "%2s", magic);
    
    // コメント行をスキップ
    int c;
    while ((c = fgetc(fp)) == '#') {
        while (fgetc(fp) != '\n');
    }
    ungetc(c, fp);
    
    int width, height, maxval;
    fscanf(fp, "%d %d %d", &width, &height, &maxval);
    fgetc(fp); // 改行読み飛ばし
    
    if (width * height != IMAGE_SIZE) {
        printf("画像サイズが不正です: %dx%d (期待値: 28x28)\n", width, height);
        fclose(fp);
        return NULL;
    }
    
    int* data = (int*)malloc(sizeof(int) * IMAGE_SIZE);
    if (!data) {
        fclose(fp);
        return NULL;
    }
    
    if (strcmp(magic, "P5") == 0) {
        for (int i = 0; i < IMAGE_SIZE; i++) {
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

// Softmax関数（オーバーフロー対策付き）
double* softmax(double *input, int size) {
    double *output = (double*)malloc(size * sizeof(double));
    if (!output) return NULL;
    
    // 最大値を求める（オーバーフロー対策）
    double max_val = input[0];
    for (int i = 1; i < size; i++) {
        if (input[i] > max_val) max_val = input[i];
    }
    
    // exp計算と合計値計算
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        output[i] = exp(input[i] - max_val);
        sum += output[i];
    }
    
    // 正規化
    if (sum > 1e-10) {
        for (int i = 0; i < size; i++) {
            output[i] /= sum;
        }
    }
    
    return output;
}

// 順伝播（is_training: 学習時はtrue、推論時はfalse）
double* forward_pass(NeuralNetwork *nn, double *input, int is_training) {
    // 入力層にデータをセット
    for (int i = 0; i < nn->image_size; i++) {
        nn->layer_output[0][i] = input[i];
    }
    
    // 各層の計算
    for (int l = 0; l < nn->layer_count - 1; l++) {
        int input_size = nn->image_size ;
        int output_size = (l == nn->layer_count - 2) ? nn->output_size : nn->image_size;
        
        for (int j = 0; j < output_size; j++) {
            double sum = nn->bias[l + 1][j];
            
            for (int i = 0; i < input_size; i++) {
                sum += nn->layer_output[l][i] * nn->synapse[l][i][j];
            }
            
            // 中間層はReLU、出力層は線形
            if (l < nn->layer_count - 2) {
                nn->layer_output[l + 1][j] = (sum > 0.0) ? sum : 0.0;
                
                // 学習時のみドロップアウト適用
                if (is_training && (rand() / (double)RAND_MAX) < DROPOUT_RATE) {
                    nn->layer_output[l + 1][j] = 0.0;
                } else if (is_training) {
                    // インバースドロップアウト
                    nn->layer_output[l + 1][j] /= (1.0 - DROPOUT_RATE);
                }
            } else {
                nn->layer_output[l + 1][j] = sum;
            }
        }
    }
    
    // 出力層にSoftmax適用
    return softmax(nn->layer_output[nn->layer_count - 1], nn->output_size);
}

// バックプロパゲーション
void backward_pass(NeuralNetwork *nn, int target, double *output) {
    double *delta = (double*)calloc(nn->image_size, sizeof(double));
    
    // 出力層の誤差計算
    for (int i = 0; i < nn->output_size; i++) {
        double target_value = (i == target) ? 1.0 : 0.0;
        double error = output[i] - target_value;
        nn->total_bias[nn->layer_count - 1][i] += error;
        
        for (int j = 0; j < nn->image_size; j++) {
            nn->total_synapse[nn->layer_count - 2][j][i] += 
                nn->layer_output[nn->layer_count - 2][j] * error;
        }
    }
    
    // 隠れ層の誤差逆伝播
    for (int l = nn->layer_count - 2; l >= 0; l--) {
        for (int i = 0; i < nn->image_size; i++) {
            double sum = 0.0;
            
            if (l == nn->layer_count - 2) {
                // 出力層からの誤差
                for (int j = 0; j < nn->output_size; j++) {
                    double target_value = (j == target) ? 1.0 : 0.0;
                    sum += nn->synapse[l][i][j] * (output[j] - target_value);
                }
            } else {
                // それ以外の層からの誤差
                for (int j = 0; j < nn->image_size; j++) {
                    sum += nn->synapse[l][i][j] * delta[j];
                }
            }
            
            // ReLUの微分
            if (nn->layer_output[l][i] > 0.0) {
                delta[i] = sum;
                nn->total_bias[l][i] += sum;
                
                if (l > 0) {
                    for (int j = 0; j < nn->image_size; j++) {
                        nn->total_synapse[l - 1][j][i] += 
                            nn->layer_output[l - 1][j] * sum;
                    }
                }
            } else {
                delta[i] = 0.0;
            }
        }
    }
    
    free(delta);
}

// パラメータ更新
void update_parameters(NeuralNetwork *nn) {
    const double threshold = 1.0;  // 勾配クリッピング
    
    for (int l = 0; l < nn->layer_count; l++) {
        int input_size = (l == 0) ? nn->image_size : nn->image_size;
        int output_size = (l == nn->layer_count - 1) ? nn->output_size : nn->image_size;
        
        // バイアス更新
        for (int i = 0; i < output_size; i++) {
            double gradient = nn->total_bias[l][i] / nn->batch_size;
            if (gradient > threshold) gradient = threshold;
            if (gradient < -threshold) gradient = -threshold;
            
            nn->bias[l][i] -= nn->learning_rate * gradient;
            nn->total_bias[l][i] = 0.0;
        }
        
        // 重み更新
        for (int i = 0; i < input_size; i++) {
            for (int j = 0; j < output_size; j++) {
                double gradient = nn->total_synapse[l][i][j] / nn->batch_size;
                if (gradient > threshold) gradient = threshold;
                if (gradient < -threshold) gradient = -threshold;
                
                nn->synapse[l][i][j] -= nn->learning_rate * gradient;
                nn->total_synapse[l][i][j] = 0.0;
            }
        }
    }
}

// 画像判定（推論専用 - 全ニューロン活性化）
int judge_image(NeuralNetwork *nn, const char *filename) {
    int *pixels = read_pgm(filename);
    if (!pixels) return ERROR_FILE_NOT_FOUND;
    
    // 正規化
    double *normalized_input = (double*)malloc(nn->image_size * sizeof(double));
    for (int i = 0; i < nn->image_size; i++) {
        normalized_input[i] = (double)pixels[i] / 255.0;
    }
    
    // 推論実行（ドロップアウトなし）
    double *output = forward_pass(nn, normalized_input, 0);
    
    // 結果表示
    int predicted_class = 0;
    for (int i = 0; i < nn->output_size; i++) {
        printf("判定値[%d]: %lf\n", i, output[i]);
        if (output[i] > output[predicted_class]) {
            predicted_class = i;
        }
    }
    
    printf("判定結果: %d\n", predicted_class);
    
    free(pixels);
    free(normalized_input);
    free(output);
    
    return predicted_class;
}

// ラベルファイル読み込み関数
int* read_labels(const char *filename, int *count) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        printf("ラベルファイルを開けません: %s\n", filename);
        return NULL;
    }

    // 行数カウント
    int label_count = 0;
    char buffer[128]; // バッファサイズ拡大
    while (fgets(buffer, sizeof(buffer), fp) != NULL) {
        label_count++;
    }
    rewind(fp);

    if (label_count == 0) {
        fclose(fp);
        *count = 0;
        return NULL;
    }

    int *labels = (int*)malloc(label_count * sizeof(int));
    if (!labels) {
        fclose(fp);
        return NULL;
    }

    for (int i = 0; i < label_count; i++) {
        if (fgets(buffer, sizeof(buffer), fp) != NULL) {
            char *endptr;
            errno = 0;
            int value = (int)strtol(buffer, &endptr, 10);
            if (errno != 0 || endptr == buffer) {
                labels[i] = 0; // エラー時は0
            } else {
                labels[i] = value;
            }
        } else {
            labels[i] = 0;
        }
    }

    fclose(fp);
    *count = label_count;
    return labels;
}

// 学習
int train_network(NeuralNetwork *nn, const char *folder_path, const char *label_file) {
    DIR *dir = opendir(folder_path);
    if (!dir) {
        printf("フォルダを開けません: %s\n", folder_path);
        return ERROR_FILE_NOT_FOUND;
    }
    
    // ラベル読み込み
    int label_count = 0;
    int *labels = read_labels(label_file, &label_count);
    if (!labels) {
        printf("ラベルファイルを読み込めませんでした: %s\n", label_file);
        closedir(dir);
        return ERROR_FILE_NOT_FOUND;
    }
    
    // PGMファイル一覧を取得
    char **file_list = (char**)malloc(MAX_FILES * sizeof(char*));
    for (int i = 0; i < MAX_FILES; i++) {
        file_list[i] = (char*)malloc(MAX_FILENAME * sizeof(char));
    }
    
    struct dirent *entry;
    int file_count = 0;
    while ((entry = readdir(dir)) != NULL && file_count < MAX_FILES) {
        if (strstr(entry->d_name, ".pgm")) {
            strcpy(file_list[file_count], entry->d_name);
            file_count++;
        }
    }
    
    closedir(dir);
    
    // ファイル数とラベル数の確認
    if (file_count != label_count) {
        printf("警告: ファイル数(%d)とラベル数(%d)が一致しません\n", file_count, label_count);
        // 少ない方に合わせる
        file_count = (file_count < label_count) ? file_count : label_count;
    }
    
    // ファイルをソート
    for (int i = 0; i < file_count - 1; i++) {
        for (int j = i + 1; j < file_count; j++) {
            if (strcmp(file_list[i], file_list[j]) > 0) {
                char temp[MAX_FILENAME];
                strcpy(temp, file_list[i]);
                strcpy(file_list[i], file_list[j]);
                strcpy(file_list[j], temp);
                
                // ラベルも同じ順序で入れ替え
                int temp_label = labels[i];
                labels[i] = labels[j];
                labels[j] = temp_label;
            }
        }
    }
    
    // 学習実行
    nn->time_step = 0;
    for (int i = 0; i < file_count; i++) {
        char filepath[MAX_PATH];
        snprintf(filepath, sizeof(filepath), "%s/%s", folder_path, file_list[i]);
        int *pixels = read_pgm(filepath);
        if (!pixels) continue;
        
        // 正規化
        double *normalized_input = (double*)malloc(nn->image_size * sizeof(double));
        for (int j = 0; j < nn->image_size; j++) {
            normalized_input[j] = (double)pixels[j] / 255.0;
        }
        
        // ラベルファイルから正解ラベルを取得
        int target = labels[i];
        
        // 順伝播（学習モード）
        double *output = forward_pass(nn, normalized_input, 1);
        
        // 損失計算
        if (output[target] > 1e-10) {
            nn->total_cost -= log(output[target]);
        }
        
        // 正解判定
        int predicted = 0;
        printf("ファイル: %s\n", filepath);
        for (int j = 1; j < nn->output_size; j++) {
            if (output[j] > output[predicted]) predicted = j;
            printf("判定値[%d]: %lf\n", j, output[j]);

        }
        printf("判定結果: %d (正解: %d)\n", predicted, target);
        


        
        if (predicted == target) nn->correct_count++;
        
        // バックプロパゲーション
        backward_pass(nn, target, output);
        nn->time_step++;
        
        // バッチ処理
        if (nn->time_step % nn->batch_size == 0) {
            update_parameters(nn);
            nn->correct_rate = (double)nn->correct_count / nn->time_step * 100.0;
        }
        
        free(pixels);
        free(normalized_input);
        free(output);
    }
    
    // メモリ解放
    for (int i = 0; i < MAX_FILES; i++) {
        free(file_list[i]);
    }
    free(file_list);
    free(labels);
    
    return file_count;
}

// ネットワークリセット
void reset_network(NeuralNetwork *nn) {
    for (int l = 0; l < nn->layer_count; l++) {
        int input_size = (l == 0) ? nn->image_size : nn->image_size;
        int output_size = (l == nn->layer_count - 1) ? nn->output_size : nn->image_size;
        
        // バイアス初期化
        for (int i = 0; i < output_size; i++) {
            nn->bias[l][i] = 0.0;
        }
        
        // 重み初期化（He初期化）
        for (int i = 0; i < input_size; i++) {
            for (int j = 0; j < output_size; j++) {
                nn->synapse[l][i][j] = randn(0.0, sqrt(2.0 / input_size));
            }
        }
    }
    
    nn->time_step = 0;
    nn->correct_count = 0;
    nn->total_cost = 0.0;
    nn->correct_rate = 0.0;
}

// ネットワーク保存
int save_network(const NeuralNetwork *nn, const char *filename) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) return ERROR_FILE_NOT_FOUND;
    
    // ヘッダー情報保存
    fwrite(&nn->layer_count, sizeof(int), 1, fp);
    fwrite(&nn->image_size, sizeof(int), 1, fp);
    fwrite(&nn->output_size, sizeof(int), 1, fp);
    fwrite(&nn->batch_size, sizeof(int), 1, fp);
    fwrite(&nn->learning_rate, sizeof(float), 1, fp);
    
    // パラメータ保存
    for (int l = 0; l < nn->layer_count; l++) {
        int input_size = (l == 0) ? nn->image_size : nn->image_size;
        int output_size = (l == nn->layer_count - 1) ? nn->output_size : nn->image_size;
        
        fwrite(nn->bias[l], sizeof(double), output_size, fp);
        
        for (int i = 0; i < input_size; i++) {
            fwrite(nn->synapse[l][i], sizeof(double), output_size, fp);
        }
    }
    
    fclose(fp);
    return SUCCESS;
}

// ネットワーク読み込み
int load_network(NeuralNetwork *nn, const char *filename) {
    printf("ファイル読み込み開始: %s\n", filename);
    
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        printf("エラー: ファイルを開けません: %s\n", filename);
        return ERROR_FILE_NOT_FOUND;
    }
    
    // ファイルサイズをチェック
    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    printf("ファイルサイズ: %ld バイト\n", file_size);
    
    // 一時的な構造体変数を使用
    int temp_layer_count, temp_image_size, temp_output_size, temp_batch_size;
    float temp_learning_rate;
    
    // ヘッダー情報読み込み（エラーチェック付き）
    if (fread(&temp_layer_count, sizeof(int), 1, fp) != 1) {
        printf("エラー: layer_count読み込みに失敗\n");
        fclose(fp);
        return ERROR_INVALID_FORMAT;
    }
    
    // 構造体の整合性チェック
    if (temp_layer_count != nn->layer_count || 
        temp_image_size != nn->image_size || 
        temp_output_size != nn->output_size) {
        printf("エラー: 構造体の構造が一致しません\n");
        fclose(fp);
        return ERROR_INVALID_FORMAT;
    }
    
    // メモリが確保されているかチェック
    for (int l = 0; l < nn->layer_count; ++l) {
    if (!nn->bias[l] || !nn->synapse[l]) {
        printf("エラー: 層 %d のメモリが確保されていません\n", l);
        fclose(fp);
        return ERROR_MEMORY_ALLOCATION;
    }
}
    
    fclose(fp);
    return SUCCESS;
}

// メイン関数

int main() {
    NeuralNetwork nn;
    int batch_size = 32;
    float learning_rate = 0.001f;
    
    // ネットワーク初期化
    if (init_neural_network(&nn, batch_size, learning_rate) != SUCCESS) {
        printf("ニューラルネットワークの初期化に失敗しました。\n");
        return 1;
    }
    
    char mode[10];
    int initialized = 0;
    
    while (1) {
        if (!initialized) {
            printf("importもしくはresetを選択してください。(import/reset): ");
            scanf("%9s", mode);
            if (strcmp(mode, "reset") == 0) {
                reset_network(&nn);
                printf("ニューラルネットワークを初期化しました。\n");
                initialized = 1;
            } else if (strcmp(mode, "import") == 0) {
                if (load_network(&nn, "savedate.param") == SUCCESS) {
                    printf("パラメータを読み込みました。\n");
                    initialized = 1;
                } else {
                    printf("パラメータファイルが見つかりません。\n");
                }
            }
            
            continue;
        }
        
        printf("モードを選択してください（judge/learn/save/reset/setting/exit）: ");
        scanf("%9s", mode);
        
        if (strcmp(mode, "judge") == 0) {
            char filename[MAX_FILENAME];
            printf("PGMファイル名を入力してください: ");
            scanf("%255s", filename);
            int result = judge_image(&nn, filename);
            

        } else if (strcmp(mode, "learn") == 0) {
            char folder_path[MAX_PATH];
            char label_file[MAX_PATH];
            
            printf("PGMファイルを含むフォルダを入力してください: ");
            scanf("%511s", folder_path);
            
            printf("ラベルファイルを入力してください: ");
            scanf("%511s", label_file);
            
            int count = train_network(&nn, folder_path, label_file);
            if (count > 0) {
                printf("エポックの損失関数: %.10f\n", nn.total_cost / (count / nn.batch_size));
                printf("エポックの正答率: %f%%\n", nn.correct_rate);
                    nn.total_cost = 0.0;
                    nn.correct_rate = 0.0;
                    nn.correct_count= 0;    
                    nn.time_step = 0;
            }
            
        } else if (strcmp(mode, "save") == 0) {
            if (save_network(&nn, "savedate.param") == SUCCESS) {
                printf("パラメータを保存しました。\n");
            } else {
                printf("保存に失敗しました。\n");
            }
            
        } else if (strcmp(mode, "reset") == 0) {
            reset_network(&nn);
            printf("ニューラルネットワークを初期化しました。\n");
            
        } else if (strcmp(mode, "setting") == 0) {
            printf("batch size: ");
            scanf("%d", &nn.batch_size);
            printf("learning rate: ");
            scanf("%f", &nn.learning_rate);
            
        } else if (strcmp(mode, "exit") == 0) {
            break;
            
        } else {
            printf("不正なモードです。\n");
        }
    }
    
    free_neural_network(&nn);
    return 0;
}

