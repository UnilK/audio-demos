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

const unsigned FRAMERATE = 44100; // this demo only works with 44100 Hz wav files.
const unsigned WSIZE = 1<<11; // window size in samples

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

vector<float> norms(vector<complex<float> > f){
    vector<float> abso(f.size());
    for(unsigned i=0; i<f.size(); i++) abso[i] = std::norm(f[i]);
    return abso;
}

vector<float> cos_cepstrum(vector<float> v, bool ignore_first = 0){

    const unsigned BINS = 32;
    const unsigned INITIAL_WIDTH = 100; // the width of the first bin in Hz

    int z = v.size();

    long double mul = 1;
    vector<std::array<float, 2> > bins;
    {
        
        long double width = INITIAL_WIDTH / ((long double)FRAMERATE / WSIZE);

        // binary search the right step size using geometrix series formula

        for(long double step = 16; step > 1e-12; step /= 2){
            auto x = mul + step;
            if(width * (1 - std::pow(x, BINS)) / (1 - x) < z) mul += step;
        }

        width /= mul;

        long double pos = 0;
        while(pos < v.size()){
            bins.push_back({(float)pos, (float)width});
            width *= mul;
            pos += width;
        }
    }

    vector<float> cepstrum(v.size(), 0.0f);

    for(auto [p, w] : bins){

        if(ignore_first && p == bins[0][0]) continue;
        
        float lw = w, rw = w * mul, sum = 0.0, warea = 0.0;
        for(int i=std::max<int>(std::ceil(p-lw), 0); i<p+rw && i<z; i++){
            if(i < p){
                float window = 0.5f + 0.5f * std::cos(M_PI * (i-p) / lw);
                sum += window * v[i];
                warea += window;
            } else {
                float window = 0.5f + 0.5f * std::cos(M_PI * (i-p) / rw);
                sum += window * v[i];
                warea += window;
            }
        }

        sum /= (warea + 1e-18);

        for(int i=std::max<int>(std::ceil(p-lw), 0); i<p+rw && i<z; i++){
            if(i < p){
                cepstrum[i] += sum * (0.5f + 0.5f * std::cos(M_PI * (i-p) / lw));
            } else {
                cepstrum[i] += sum * (0.5f + 0.5f * std::cos(M_PI * (i-p) / rw));
            }
        }
    }

    return cepstrum;
}

void generate_profile(){

    iwstream in("mic-noise.wav");

    auto audio = in.read_file();

    vector<float> buffer(WSIZE, 0.0f), sum(WSIZE / 2 + 1, 0.0f), window(WSIZE);
    for(unsigned i=0; i<WSIZE; i++){
        window[i] = 0.5f - 0.5f * std::cos(2 * M_PI * i / WSIZE);
    }

    int cnt = 0;
    for(unsigned i=0; i+WSIZE<=audio.size(); i+=WSIZE/2){
        cnt++;
        for(unsigned j=0; j<WSIZE; j++) buffer[j] = audio[i+j] * window[j];
        vector<float> profile = cos_cepstrum(norms(fft(buffer)), 1);
        for(unsigned j=0; j<profile.size(); j++) sum[j] += profile[j];
    }

    for(float &i : sum) i /= (cnt + 1e-18f);

    std::ofstream out("mic-noise-profile.txt");
    
    out << std::setprecision(10) << std::fixed;
    for(float i : sum) out << i << '\n';
}

void denoise_audio(){
    
    vector<float> profile(WSIZE / 2 + 1), audio;
    
    {
        std::ifstream in("mic-noise-profile.txt");
        for(float &i : profile) in >> i;
        for(float &i : profile) i *= 12;

        iwstream ain("input.wav");
        audio = ain.read_file();
    }

    vector<float> result(audio.size(), 0.0f), window(WSIZE), buffer(WSIZE);
    for(unsigned i=0; i<WSIZE; i++){
        window[i] = 1.0f - std::cos(2 * M_PI * i / WSIZE);
    }

    for(unsigned i=0; i+WSIZE<=audio.size(); i+=WSIZE/4){
        
        for(unsigned j=0; j<WSIZE; j++) buffer[j] = audio[i+j] * window[j];
        auto freq = fft(buffer);
        auto ceps = cos_cepstrum(norms(freq));

        for(unsigned i=0; i<freq.size(); i++){
            float d = std::max<float>(0, ceps[i] - profile[i]) / (ceps[i] + 1e-18f);
            freq[i] *= d;
        }

        buffer = inverse_fft(freq);

        for(unsigned j=0; j<WSIZE; j++) result[i+j] += buffer[j] * window[j];
    }

    for(float &i : result) i /= 6;

    owstream out("output.wav", 0x1, 1, 16, FRAMERATE);
    out.write_file(result);
}

int main(int argc, char *argv[]){

    bool action = 0;

    if(argc > 1 && std::string(argv[1]) == "profile") action = 1;

    if(action) generate_profile();
    else denoise_audio();

    return 0;
}
