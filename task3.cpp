#include <cmath>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <mpi.h>
#include <unistd.h>
#include <fstream>

#define A1 0
#define A2 2
#define B1 0
#define B2 1
#define EPS 0.000001
#define DEBUG 0
#define M 160
#define N 160

using namespace std;

double u(double x, double y)
{
    return 1 + cos(M_PI * x * y);
}

double psi_r(double x, double y)
{
    return -M_PI * y * sin(M_PI * x * y) * (4 + x + y) + 1 + cos(M_PI * x * y);
}

double psi_l(double x, double y)
{
    return M_PI * y * sin(M_PI * x * y) * (4 + x + y);
}

double psi_t(double x, double y)
{
    return -M_PI * x * sin(M_PI * x * y) * (4 + x + y) + 1 + cos(M_PI * x * y);
}

double psi_b(double x, double y)
{
    return M_PI * x * sin(M_PI * x * y) * (4 + x + y);
}

double k(double x, double y)
{
    return 4 + x + y;
}

double F(double x, double y)
{
    return M_PI * (x + y) * sin(M_PI * x * y) + M_PI * M_PI * (4 + x + y) * cos(M_PI * x * y) * (x * x + y * y);
}

void diff(double *diff, double *w1, double *w2, int size_x, int size_y)
{
    int i, j;
    # pragma omp parallel for private(i, j)
    for(i = 1; i <= size_x + 1; i++)
    {
        for (j = 0; j <= size_y + 1; j++)
        {
            diff[i * (size_y + 2) + j] = w1[i * (size_y + 2) + j] - w2[i * (size_y + 2) + j];
        }
    }
}

 double ro_i(int i, int size_x, int size_y)
{
    if (i == 1 || i == size_x)
        return 0.5;
    else
        return 1;
}

double ro_j(int j, int size_x, int size_y)
{
    if (j == 1 || j == size_y)
        return 0.5;
    else
        return 1;
}

double vector_dot(double *w0, double *w1, int size_x, int size_y, double hx, double hy)
{
    double res = 0;
    for(int i = 0; i <= size_x + 1; i++)
    {
        for (int j = 0; j <= size_y + 1; j++)
        {
            if(i != 0 && j != 0 && i != size_x + 1 && j != size_y + 1)
                res += ro_i(i, size_x, size_y) * ro_j(j, size_x, size_y) * w0[i * (size_y + 2) + j] * w1[i * (size_y + 2) + j];
        }
    }
    return res * hx * hy;
}

double vector_norm(double *w, int size_x, int size_y, double hx, double hy)
{
    return sqrt(vector_dot(w, w, size_x, size_y, hx, hy));
}


void getB(double *B, int size_x, int size_y, double hx, double hy, double x_start, double y_start)
{
    int i, j;
    # pragma omp parallel for private(i, j)
    for(i = 1; i <= size_x; i++)
        for (j = 1; j <= size_y; j++)
        {
            B[i * (size_y + 2) + j] = F(x_start + (i - 1) * hx, y_start + (j - 1) * hy);
        }
    if (abs(x_start - A1) < EPS)
        for (int j = 1; j <= size_y; j++)
            B[j] = F(x_start, y_start + (j - 1) * hy) + 2/hx * psi_l(x_start, y_start + (j - 1) * hy);
    if (abs(x_start + (size_x + 1)*hx - A2) < EPS)
        for (int j = 1; j <= size_y; j++)
            B[(size_x + 1) * (size_y + 2) + j] = F(x_start + (size_x + 1)*hx, y_start + (j - 1) * hy) + 2/hx * psi_r(x_start + (size_x + 1) * hx, y_start + (j - 1) * hy);
    if (abs(y_start + (size_y + 1)*hy - B2) < EPS)
        for(int i = 1; i <= size_x; i++)
            B[i * (size_y + 2) + size_y + 1] = F(x_start + (i - 1)*hx, y_start + (size_y + 1) * hy) + 2/ hy * psi_t(x_start + (i - 1)*hx, y_start + (size_y + 1) * hy);
    if (abs(y_start - B1) < EPS)
        for(int i = 1; i <= M; i++)
            B[i * (size_y + 2)] = F(x_start + (i - 1)*hx, y_start + (size_y + 1) * hy) + 2/ hy * psi_b(x_start + (i - 1)*hx, y_start + (size_y + 1) * hy);
    
}

void getAw(double *A, double *w, int size_x, int size_y, double hx, double hy, double x_start, double y_start)
{
    for(int i = 0; i <= size_x + 1; i++)
    {
        for (int j = 0; j <= size_y + 1; j++)
        {
            
            if(i == 0 || j == 0 || i == size_x + 1 || j == size_y + 1)
            {
                if (abs(x_start - A1) < EPS)
                {
                    double bw = k(x_start + i * hx, y_start + (j + 0.5) * hy) * (w[i * (size_y + 2) + j + 1] - w[i * (size_y + 2) + j]) / hy
                    - k(x_start + i * hx, y_start + (j - 0.5) * hy) * (w[i * (size_y + 2) + j] - w[i * (size_y + 2) + j - 1]) / hy;
                    A[i * (size_y + 2) + j] = -2/hx * k(x_start + 0.5 * hx, y_start + j * hy) * (w[(size_y + 2) + j] - w[j]) / hx - bw/hy;
                }
                if (abs(x_start + i*hx - A2) < EPS)
                {
                    double bw = k(x_start + i * hx, y_start + (j + 0.5) * hy) * (w[i * (size_y + 2) + j + 1] - w[i * (size_y + 2) + j]) / hy
                    - k(x_start + i * hx, y_start + (j - 0.5) * hy) * (w[i * (size_y + 2) + j] - w[i * (size_y + 2) + j - 1]) / hy;
                    A[i * (size_y + 2) + j] = 2/hx *k(x_start + (size_x + 0.5) * hx, y_start + j * hy) * (w[(size_x + 1) * (size_y + 2) + j] - w[size_x* (size_y + 2) + j]) / hx
                    + 2/hx * w[(size_x + 1) * (size_y + 2) + j] - bw/hy;
                }
                if (abs(y_start + j*hy - B2) < EPS)
                {
                    double aw = k(x_start + (i + 0.5) * hx, y_start + j * hy)*(w[(i + 1) * (size_y + 2) + j] - w[i * (size_y + 2) + j]) / hx
                    - k(x_start + (i - 0.5) * hx, y_start + j * hy) * (w[i * (size_y + 2) + j] - w[(i - 1) * (size_y + 2) + j]) / hx;
                    A[i * (size_y + 2) + j] = -2/hy * k(x_start + i * hx, y_start + (j - 0.5) * hy) * (w[i * (size_y+ 2) + j] - w[i * (size_y + 2) + j - 1]) / hy
                    + 2/hy*w[i * (size_y + 2) + j] - aw/hx;
                }
                if (abs(y_start - B1) < EPS)
                {
                    double aw = k(x_start + (i + 0.5) * hx, y_start + j * hy)*(w[(i + 1) * (size_y + 2) + j] - w[i * (size_y + 2) + j]) / hx
                    - k(x_start + (i - 0.5) * hx, y_start + j * hy) * (w[i * (size_y + 2) + j] - w[(i - 1) * (size_y + 2) + j]) / hx;
                    A[i * (size_y + 2) + j] =  -2/hy * k(x_start + i * hx, y_start + (j - 0.5) * hy) * (w[i * (size_y + 2) + j] - w[i * (size_y + 2) + j - 1]) / hy - aw/hx;
                }
                    
            }
            else
            {
                double aw = k(x_start + (i + 0.5) * hx, y_start + j * hy)*(w[(i + 1) * (size_y + 2) + j] - w[i * (size_y + 2) + j]) / hx
                - k(x_start + (i - 0.5) * hx, y_start + j * hy) * (w[i * (size_y + 2) + j] - w[(i - 1) * (size_y + 2) + j]) / hx;
                double bw = k(x_start + i * hx, y_start + (j + 0.5) * hy) * (w[i * (size_y + 2) + j + 1] - w[i * (size_y + 2) + j]) / hy
                - k(x_start + i * hx, y_start + (j - 0.5) * hy) * (w[i * (size_y + 2) + j] - w[i * (size_y + 2) + j - 1]) / hy;
                A[i * (size_y + 2) + j] = -aw/hx - bw/hy;
            }
        }
    }
}

int main(int argc, char **argv)
{
    int rank;
    int commSize;
    int neighbour_rank;
    int size[2] = {0};

    double hx = 2.0 / M;
    double hy = 1.0 / N;
    double eps_global = 1000.0;
    
    MPI_Status status;
    MPI_Request request;
    MPI_Init(&argc, &argv);
    MPI_Comm MPI_COMM_CART;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);
    MPI_Dims_create(commSize, 2, size);
    
    int periods[2] = {0};
    int coords[2];
    int neighbour_coords[2];
    
    MPI_Cart_create(MPI_COMM_WORLD, 2, size, periods, true, &MPI_COMM_CART);
    MPI_Cart_coords(MPI_COMM_CART, rank, 2, coords);
    
    double start_time = MPI_Wtime();
    
    int x_shift, x_size;
    int y_shift, y_size;
    int t = 0;
    int n_it = 0;

    if (M % size[0] == 0)
    {
        x_size = M / size[0];
        x_shift = coords[0] * (M / size[0]);
    }
    else
    {
        if (coords[0] < (M % size[0]))
        {
            x_size = 1 + M / size[0];
            x_shift = coords[0] + coords[0] * (M / size[0]);
        }
        else
        {
            x_size = M / size[0];
            x_shift = M % size[0] + coords[0] * (M / size[0]);
        }
    }

    if (N % size[1] == 0)
    {
        y_size = N / size[1];
        y_shift = coords[1] * (N / size[1]);
    }
    else
    {
        if (coords[1] < (N % size[1]))
        {
            y_size = 1 + N / size[1];
            y_shift = coords[1] + coords[1] * (N / size[1]);
        }
        else
        {
            y_size = N / size[1];
            y_shift = N % size[1] + coords[1] * (N / size[1]);
        }
    }
    
    double tau;
    double eps_local;
    
    double *top_s = new double[x_size];
    double *top_r = new double[x_size];
    double *bottom_s = new double[x_size];
    double *bottom_r = new double[x_size];
    double *left_s = new double[y_size];
    double *left_r = new double[y_size];
    double *right_s = new double[y_size];
    double *right_r = new double[y_size];

    double *w = new double [(x_size + 2) * (y_size + 2)];
    double *w_prev = new double [(x_size + 2) * (y_size + 2)];
    double *B = new double [(x_size + 2) * (y_size + 2)];
    double *Aw = new double [(x_size + 2) * (y_size + 2)];
    double *r = new double [(x_size + 2) * (y_size + 2)];
    double *Ar = new double [(x_size + 2) * (y_size + 2)];
    double *w_err = new double [(x_size + 2) * (y_size + 2)];

    int i, j;
    
    # pragma omp parallel for private(i, j)
    for (i = 0; i <= x_size + 1; i++)
    {
        for (j = 0; j <= y_size + 1; j++)
        {
            w[i * (y_size + 2) + j] = 0;
        }
    }
    
    getB(B, x_size, y_size, hx, hy, A1 + x_shift * hx, B1 + y_shift * hy);

    while (eps_global > EPS)
    {
        n_it++;
        
        # pragma omp parallel for private(i, j)
        for (i = 0; i <= x_size + 1; i++)
        {
            for (j = 0; j <= y_size + 1; j++)
            {
                w_prev[i * (y_size + 2) + j] = w[i * (y_size + 2) + j];
            }
        }


        if (size[1] > 1)
        {
            for (int i = 1; i <= x_size; i++)
            {
                top_s[i - 1] = w[i * (y_size + 2) + y_size];
            }
            if (coords[1] != (size[1] - 1))
            {
                neighbour_coords[0] = coords[0];
                neighbour_coords[1] = coords[1] + 1;
                MPI_Cart_rank(MPI_COMM_CART, neighbour_coords, &neighbour_rank);
                MPI_Isend(top_s, x_size, MPI_DOUBLE, neighbour_rank, t, MPI_COMM_CART, &request);
            }
        }

        if (size[1] > 1)
        {
            for (int i = 1; i <= x_size; i++)
            {
                    bottom_s[i - 1] = w[i * (y_size + 2) + 1];
            }

            if (coords[1] != 0)
            {
                neighbour_coords[0] = coords[0];
                neighbour_coords[1] = coords[1] - 1;
                MPI_Cart_rank(MPI_COMM_CART, neighbour_coords, &neighbour_rank);
                MPI_Isend(bottom_s, x_size, MPI_DOUBLE, neighbour_rank, t, MPI_COMM_CART, &request);
            }
        }

        if (size[0] > 1)
        {
            if (coords[0] != 0)
            {
                for (int j = 1; j <= y_size; j++)
                {
                    left_s[j - 1] = w[(y_size + 2) + j];
                }
                neighbour_coords[0] = coords[0] - 1;
                neighbour_coords[1] = coords[1];
                MPI_Cart_rank(MPI_COMM_CART, neighbour_coords, &neighbour_rank);
                MPI_Isend(left_s, y_size, MPI_DOUBLE, neighbour_rank, t, MPI_COMM_CART, &request);
            }
        }

        if (size[0] > 1)
        {
            if (coords[0] != (size[0] - 1))
            {
                for (int j = 1; j <= y_size; j++)
                {
                    right_s[j - 1] = w[x_size *(y_size + 2) + j];
                }
                neighbour_coords[0] = coords[0] + 1;
                neighbour_coords[1] = coords[1];
                MPI_Cart_rank(MPI_COMM_CART, neighbour_coords, &neighbour_rank);
                MPI_Isend(right_s, y_size, MPI_DOUBLE, neighbour_rank, t, MPI_COMM_CART, &request);
            }
        }

        if ((size[1] > 1 && coords[1] == size[1] - 1) || size[1] == 1)
        {
            for (int i = 1; i <= x_size; i++)
            {
                w[i * (y_size + 2) + y_size + 1] = u(A1 + (x_shift + i) * hx, B2);
            }
        } else {
            neighbour_coords[0] = coords[0];
            neighbour_coords[1] = coords[1] + 1;
            MPI_Cart_rank(MPI_COMM_CART, neighbour_coords, &neighbour_rank);
            MPI_Recv(top_r, x_size, MPI_DOUBLE, neighbour_rank, t, MPI_COMM_CART, &status);
            for (int i = 1; i <= x_size; i++)
            {
                w[i * (y_size + 2) + y_size + 1] = top_r[i - 1];
            }
        }

        if((coords[1] == 0 && size[1] > 1) || size[1] == 1)
        {
            for (int i = 1; i <= x_size; i++)
            {
                w[i * (y_size + 2)] = u(A1 + (x_shift + i) * hx, B1);
            }
        } else {
            neighbour_coords[0] = coords[0];
            neighbour_coords[1] = coords[1] - 1;
            MPI_Cart_rank(MPI_COMM_CART, neighbour_coords, &neighbour_rank);
            MPI_Recv(bottom_r, x_size, MPI_DOUBLE, neighbour_rank, t, MPI_COMM_CART, &status);
            for (i = 1; i <= x_size; i++)
            {
                w[i * (y_size + 2)] = bottom_r[i - 1];
            }
        }

        if ((size[0] > 1 && coords[0] == 0) || size[0] == 1)
        {
            for (int j = 1; j <= y_size; j++)
            {
                w[j] = u(A1, B1 + (y_shift + j) * hy);
            }
        } else {
            neighbour_coords[0] = coords[0] - 1;
            neighbour_coords[1] = coords[1];
            MPI_Cart_rank(MPI_COMM_CART, neighbour_coords, &neighbour_rank);
            MPI_Recv(left_r, y_size, MPI_DOUBLE, neighbour_rank, t, MPI_COMM_CART, &status);
            for (int j = 1; j <= y_size; j++)
            {
                w[j] = left_r[j - 1];
            }
        }

        if ((size[0] > 1 && coords[0] == (size[0] - 1)) || size[0] == 1)
        {
            for (j = 1; j <= y_size; j++)
            {
                w[(x_size + 1) * (y_size + 2) + j] = u(A2, B1 + (y_shift + j) * hy);
            }
        } else {
            neighbour_coords[0] = coords[0] + 1;
            neighbour_coords[1] = coords[1];
            MPI_Cart_rank(MPI_COMM_CART, neighbour_coords, &neighbour_rank);
            MPI_Recv(right_r, y_size, MPI_DOUBLE, neighbour_rank, t, MPI_COMM_CART, &status);
            for (int j = 1; j <= y_size; j++)
            {
                w[(x_size + 1) * (y_size + 2) + j] = right_r[j - 1];
            }
        }

        getAw(Aw, w, x_size, y_size, hx, hy, A1 + x_shift * hx, B1 + y_shift * hy);
        diff(r, Aw, B, x_size, y_size);
        getAw(Ar, r, x_size, y_size, hx, hy, A1 + x_shift * hx, B1 + y_shift * hy);
        tau = vector_dot(Ar, r, x_size, y_size, hx, hy) / vector_dot(Ar, Ar, x_size, y_size, hx, hy);
        
        # pragma omp parallel for private(i, j)
        for (i = 1; i <= x_size; i++)
        {
            for (j = 1; j <= y_size; j++)
            {
                w[i * (y_size + 2) + j] = w[i*(y_size + 2) + j] - tau * r[i * (y_size + 2) + j];
            }
        }

        diff(w_err, w, w_prev, x_size, y_size);
        eps_local = vector_norm(w_err, x_size, y_size, hx, hy);

        MPI_Allreduce(&eps_local, &eps_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    }

    MPI_Barrier(MPI_COMM_WORLD);
    
    double end_time = MPI_Wtime();

    
    if (rank == 0)
    {
        cout << "Time: " << end_time - start_time << endl;
        cout << "Iterations: " << n_it << endl;
    }

    for (int i = 1; i <= x_size; i++)
    {
        for (int j = 1; j <= y_size; j++)
            cout << x_shift + i - 1<< " " << y_shift + j - 1 << " " << w[i*(y_size + 2) + j] << endl;
        cout << endl;
    }

    delete[] w;
    delete[] w_prev;
    delete[] B;
    delete[] r;
    delete[] Ar;
    delete[] Aw;
    delete[] w_err;
    delete[] top_s;
    delete[] top_r;
    delete[] bottom_s;
    delete[] bottom_r;
    delete[] left_s;
    delete[] left_r;
    delete[] right_s;
    delete[] right_r;

    MPI_Finalize();
    return 0;
}
