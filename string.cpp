// The code requires armadillo library http://arma.sourceforge.net/

// The code uses the string method with m=infinity and m^*=1, and it can be used to produce Fig.4 of the article.

// compile:
// $ g++ MF_string_P.cpp -o MF_string_P -DARMA_DONT_USE_WRAPPER -lopenblas -llapack
// input:
// - d = input dimension;
// - P = dataset size;
// - epochs = number of epochs for GD;
// - rate = learning rate for GD;
// - slices = number of points in the string;
// - epochs_print_n = number of epoch to output;
// - seed = random seed;
// output:
// - "epochs_print_n" files in "directory", one for each epoch taken in exponential scale. The files have a row per slice and 5 column: 
//   1) slice index, 2) training loss, 3) generalization loss, 4) gradient norm;
// - dataset matrix X;
// - A matrix at the end of the training for each slice in the string.


#include <iostream>
#include <armadillo>
#include <fstream>
using namespace std;

void interpolate(double *line, int slices, arma::Mat<double>* As, arma::Mat<double>& A, double x);
double updateA(const arma::Mat<double>& At, const arma::Mat<double>& Astar, arma::Mat<double>& grad, const arma::Mat<double>& x, int P);
double pow_int(double x, int p);
std::vector<double> linspace(double start, double end, int num);

int main(int argc, const char **argv) {
    int seed;

    int d, epochs, P, epochs_print_n;
    int e, P_idx, idx, p, epochs_print_idx;
    int slice, slices;
    double alpha, tmp_double,rate;
    double *loss_g, *loss_t, *grad_norm, *line, *line0;
    arma::Mat<double> x,xx,tmp_mat, grad, Astar, A0, At;
    vector<int>::iterator ip; 
    vector<int> epochs_print;
    char fout_name[100];
    ofstream fout;

    // specify output folder
    char directory[100]="";
    
    // load input from terminal or, in alternative, use default options
    if(argc==1){
        d = 8; 
        P = 9;
        epochs = 1000000; 
        rate = 0.003; 
        slices = 100;
        epochs_print_n = 20;
        seed = 1;
    }else{
        d = atoi(argv[1]); 
        P = atoi(argv[2]);
        epochs = atoi(argv[3]); 
        rate = atof(argv[4]); 
        slices = atoi(argv[5]);
        epochs_print_n = atoi(argv[6]);
        seed = atoi(argv[7]);
    }
    // Initialize the random generator
    arma::arma_rng::set_seed(seed); // set random seed

    loss_g = (double*)malloc((slices)*sizeof(double));
    loss_t = (double*)malloc((slices)*sizeof(double));
    grad_norm = (double*)malloc((slices)*sizeof(double));
    line = (double*)malloc((slices)*sizeof(double));
    line0 = (double*)malloc((slices)*sizeof(double));
    for (slice = 0; slice<slices; slice++) line[slice] = double(slice)/double(slices-1);
    for (slice = 0; slice<slices; slice++) line0[slice] = double(slice)/double(slices-1);

    // generate and print database
    x = arma::randn(P,d);
    sprintf(fout_name, "%sPR_cpp_string_d%d_P%d_minf_lr%.2e_seed%d_X.txt", directory, d, P, rate, seed); fout.open(fout_name);
    fout.precision(17);
    for(int i=0;i<P;i++){
        for(int j=0;j<d;j++){
            fout << x(i,j) << "\t";
        }
        fout << "\n";
    }
    fout.close();

    Astar = arma::zeros(d,d); Astar(0,0) = 1.;
    A0 = arma::eye(d,d), At = arma::eye(d,d);

    arma::Mat<double> As[slices], Aaux[slices];
    for (slice = 0; slice<slices; slice++) As[slice] = double(slice)/double(slices-1)*Astar + double(slices-slice-1)/double(slices-1)*A0;

    // generate list of times to print
    vector<double> tmp_vec = (linspace(log(1),log(epochs),epochs_print_n));
    for (idx=0;idx<epochs_print_n;idx++) epochs_print.push_back(int(exp(tmp_vec[idx]))); 
    ip = std::unique(epochs_print.begin(), epochs_print.end()); 
    epochs_print.resize(std::distance(epochs_print.begin(), ip)); epochs_print_n = epochs_print.size();
    sort(epochs_print.begin(), epochs_print.end());
    epochs_print_idx = 0;


    // initialization
    for(slice=0;slice<slices;slice++){
        grad = arma::zeros(d,d); At = As[slice];
        loss_t[slice] = updateA(At, Astar, grad, x, P); 
        grad = grad/P; loss_t[slice] /= P;
        loss_g[slice] = 0.25*(pow_int(trace(As[slice]-Astar),2)+2.0*trace((As[slice]-Astar)*(As[slice]-Astar)));
        grad_norm[slice] = norm(grad,"fro");
        if (isnan(loss_t[slice]) || isnan(loss_g[slice]) || isnan(grad_norm[slice])){ perror("NaN reached in definition"); return 17; }
    }
    grad = arma::zeros(d,d); At = arma::zeros(d,d); 

    // start gradient descent dynamics
    for(e=1;e<=epochs;e++){
        // update string
        for(slice=1;slice<slices-1;slice++){
            for(idx=0;idx<d;idx++) for(int idx2=0;idx2<d;idx2++){ grad[idx,idx2] = 0.; }
            At = As[slice];
            loss_t[slice] = updateA(At, Astar, grad, x, P); 
            grad = grad/P; loss_t[slice] /= P;
            loss_g[slice] = 0.25*(pow_int(trace(At-Astar),2)+2.0*trace((At-Astar)*(At-Astar)));
            grad_norm[slice] = norm(grad,"fro");

            As[slice] = At + rate*grad;
            
            if (isnan(loss_t[slice]) || isnan(loss_g[slice]) || isnan(grad_norm[slice])){ perror("NaN reached in evolution"); return 17; }
        }

        // reparametrize string
        for (slice = 1; slice<slices; slice++) line[slice] = norm(As[slice-1]-As[slice],"fro");
        for (slice = 1; slice<slices; slice++) line[slice] += line[slice-1];
        for (slice = 1; slice<slices; slice++) line[slice] /= line[slices-1];
        for(slice=0;slice<slices;slice++){
            interpolate( line, slices, As, Aaux[slice], line0[slice]);
        }
        for(slice=1;slice<slices-1;slice++) As[slice] = Aaux[slice];

        // output if in output times list
        if(e==epochs_print[epochs_print_idx]){
            cout << e << " ";

            // print string at current time
            sprintf(fout_name, "%sPR_cpp_string_d%d_P%d_minf_lr%.2e_seed%d_%d.txt", directory, d, P, rate, seed, e); fout.open(fout_name);
            fout.precision(17);
            for(slice=0;slice<slices;slice++){ 
                if (isnan(loss_t[slice]) || isnan(loss_g[slice]) || isnan(grad_norm[slice])){ perror("NaN reached in printing"); return 17; }
                fout << slice << "\t" << loss_t[slice] << "\t" << loss_g[slice] << "\t" << grad_norm[slice] << "\n";
            }
            fout.close();

            epochs_print_idx++;

            // print A files at current time
            for (slice = 0; slice<slices; slice++){
                sprintf(fout_name, "%sPR_cpp_string_d%d_P%d_minf_lr%.2e_seed%d_A%d.txt", directory, d, P, rate, seed, slice); fout.open(fout_name);
                fout.precision(17);
                for(int i=0;i<d;i++){
                    for(int j=0;j<d;j++){
                        fout << As[slice](i,j) << "\t";
                    }
                    fout << "\n";
                }
                fout.close();
            } 
        }
    }
    printf("\nDone!\n");
    return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// one step gradient descent
double updateA(const arma::Mat<double>& At, const arma::Mat<double>& Astar, arma::Mat<double>& grad, const arma::Mat<double>& x, int P){
    arma::Mat<double> xx,tmp_mat;
    double tmp_double, loss_t = 0.;

    for (int p=0;p<P;p++){
        xx = x.row(p).t()*x.row(p);
        tmp_mat = At*xx; tmp_mat += tmp_mat.t();
        tmp_double = sum(sum((Astar-At)%xx));
        grad += tmp_double*tmp_mat;
        loss_t += .25*tmp_double*tmp_double;
    }
    return loss_t;
}

// 1D interpolation
void interpolate(double *line, int slices, arma::Mat<double>* As, arma::Mat<double>& A, double x){
    int i = 0;
    // find left end of interval for interpolation
    while ( x > line[i+1] ){
        i++;
        if(i>slices) perror("extrapolation not allowed");
    } 

    // gradient
    arma::Mat<double> dydx = ( As[i+1] - As[i] ) / ( line[i+1] - line[i] );                                    
    A = As[i] + dydx * ( x - line[i] );
}

// integer power
double pow_int(double x, int p) {
  if (p == 0) return 1.;
  if (p == 1) return x;
  return x * pow_int(x, p-1);
}

// generate linearly spaced vector 
std::vector<double> linspace(double start, double end, int num){
    std::vector<double> result;

    if (num == 0) return result;
    if (num == 1){
        result.push_back(start);
        return result;
    }

    double delta = (end - start) / (num - 1);
    for(int i=0; i < num-1; ++i) result.push_back(start + delta * i);
    result.push_back(end); 

    return result;
}
