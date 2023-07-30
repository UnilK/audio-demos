#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
#include <wstream/wstream.h>

using std::vector;
using std::complex;

void in_place_fft(vector<complex<float> > &v, bool inv){


    int n = v.size();

    int bits = 0;
    while(1ll<<bits < n) bits++;
    assert(1ll<<bits == n && bits < 30);
    if(bits == 0) return;

    // dynamic precalculation tables, calculated in O(n)

    static int N = -1;
    static vector<int> invbit;
    static vector<vector<complex<float> > > w;

    if(bits > N){
        
        N = bits;
        int M = bits-1;

        w.resize(N);
        invbit = vector<int>(1<<N, 0);

        w[M].resize(1<<M);
        for(int i=0; i<(1<<M); i++) w[M][i] = std::polar<double>(1.0, -M_PI * i / (1<<M));
        
        invbit[1] = 1;
        for(int b=M-1; b>=0; b--){

            int x = 1<<b, y = 1<<(M-b);
            w[b].resize(x);
            
            for(int i=0; i<y; i++){
                invbit[i] <<= 1;
                invbit[i+y] = invbit[i] | 1;
            }

            for(int i=0; i<x; i++) w[b][i] = w[b+1][2*i];
        }
    }
    
    // actual fft
    
    int shift = N - bits;

    for(int i=0; i<n; i++)
        if(i < invbit[i]>>shift) std::swap(v[i], v[invbit[i]>>shift]);

    for(int r=0, rd=1; r<bits; r++, rd*=2){
        for(int i=0; i<n; i+=2*rd){
            for(int j=i; j<i+rd; j++){
                complex<float> wv = w[r][j-i] * v[j+rd];
                v[j+rd] = v[j] - wv;
                v[j] = v[j] + wv;
            }
        }
    }

    // inverse

    if(inv){
        std::reverse(v.begin()+1, v.end());
        float invn = 1.0 / n; // accurate since n is a power of 2
        for(int i=0; i<n; i++) v[i] *= invn;
    }
}

vector<complex<float> > fft(const vector<float> &v){
    
    int n = v.size();
    int m = n/2 + 1;

    vector<complex<float> > w(n);
    for(int i=0; i<n; i++) w[i] = v[i];
    
    in_place_fft(w, 0);
    w.resize(m);
    
    return w;
}

vector<float> inverse_fft(const vector<complex<float> > &v){
    
    int m = v.size();
    int n = 2 * m - 2;

    auto w = v;
    w.resize(n);
    
    for(int i=1; i<m; i++) w[n-i] = std::conj(w[i]);

    in_place_fft(w, 1);
    
    vector<float> r(n);
    for(int i=0; i<n; i++) r[i] = w[i].real();
    
    return r;
}

int main(){

    iwstream in("input.wav");
    
    vector<float> audio = in.read_file();

    const unsigned N = 1<<12;

    vector<float> projection(audio.size(), 0.0f);
    complex<float> imag = {0.0f, 1.0f};

    vector<float> window(N), buffer(N);
    for(unsigned i=0; i<N; i++) window[i] = 1.0f - std::cos(2 * M_PI * i / N);

    for(unsigned i=0; i+N<=audio.size(); i+=N/4){
        for(unsigned j=0; j<N; j++) buffer[j] = audio[i+j] * window[j];
        auto freq = fft(buffer);
        for(auto &i : freq) i *= imag;
        buffer = inverse_fft(freq);
        for(unsigned j=0; j<N; j++) projection[i+j] += buffer[j] * window[j];
    }

    for(float &i : projection) i /= 6;

    long double correlation = 0;
    for(unsigned i=0; i<audio.size(); i++) correlation += audio[i] * projection[i];

    std::cout << "Absolute correlation: " << correlation << '\n';
    std::cout << "Per-sample correlation: " << correlation/audio.size() << '\n';

    // Because the correlation of the signals is 0, both adding and subtracting
    // them from each other amplifies the signal by a factor of sqrt(2).

    vector<float> plus = audio, minus = audio;
    for(unsigned i=0; i<audio.size(); i++) plus[i] += projection[i];
    for(unsigned i=0; i<audio.size(); i++) minus[i] -= projection[i];

    owstream output_projection("output-projection.wav", 0x1, 1, 16, 44100);
    owstream output_plus("output-plus.wav", 0x1, 1, 16, 44100);
    owstream output_minus("output-minus.wav", 0x1, 1, 16, 44100);

    output_projection.write_file(projection);
    output_plus.write_file(plus);
    output_minus.write_file(minus);

    return 0;
}
