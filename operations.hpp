/* CKKS 同态运算抽象（header-only 模板）。
 *
 * 分三层：
 *   1) 操作数存储基类 VecData<T,N>        —— 定长 N 的 std::array<T,N> + 访问器。
 *   2) 可派生操作数 Plaintext / Ciphertext —— 模拟后端下即 vec<T>；真实 CKKS 从它们
 *                                              派生带真实编码（RNS/NTT）的子类。
 *   3) 运算抽象 CkksOps<T,N>（抽象接口） + VecOps<T,N>（模拟实现）。
 *
 * 设计约束：
 *   - 定长：N 为编译期模板参数（CKKS 环维，2 的幂：32768 / 65536 …）。
 *   - 只暴露 CKKS 密文域允许的运算：ct⊕ct、ct⊕pt（明文-明文不属于同态运算，不进接口）。
 *   - rescale / modswitch 本期为桩：仅打印、不改向量数据（真实语义留待真实后端实现）。
 *
 *   VecOps<double, 65536> ops;
 *   Plaintext<double,65536>  pt;  pt.fill(2.0);
 *   Ciphertext<double,65536> ct;  ct.fill(3.0);
 *   auto s = ops.add_cp(ct, pt);   // 逐元素 5.0
 *   auto p = ops.mul_cp(ct, pt);   // 逐元素 6.0
 *   ops.rescale(p);                // 桩：仅打印
 */
#ifndef MPI_TEST_OPERATIONS_HPP
#define MPI_TEST_OPERATIONS_HPP

#include <array>
#include <cstddef>
#include <cstdio>

/* ---- 1) 定长存储基类 ---- */
template <class T, std::size_t N>
class VecData {
public:
    static constexpr std::size_t kSlots = N;

    constexpr std::size_t size() const { return N; }

    T &operator[](std::size_t i) { return data_[i]; }
    const T &operator[](std::size_t i) const { return data_[i]; }

    /* 裸指针，便于直接喂给 MPI_Send/Recv 等 C API */
    T *data() { return data_.data(); }
    const T *data() const { return data_.data(); }

    void fill(T v) { data_.fill(v); }

protected:
    std::array<T, N> data_{};
};

/* ---- 2) 可派生操作数 ---- */
template <class T, std::size_t N>
class Plaintext : public VecData<T, N> {
public:
    Plaintext() = default;

    /* 从迭代器范围构造（长度需 == N，调用方负责）*/
    template <class Iter>
    Plaintext(Iter first, Iter last) {
        for (std::size_t i = 0; first != last && i < N; ++i, ++first)
            (*this)[i] = static_cast<T>(*first);
    }
};

template <class T, std::size_t N>
class Ciphertext : public VecData<T, N> {
public:
    Ciphertext() = default;

    int  level() const { return level_; }
    void level(int l) { level_ = l; }

private:
    int level_ = 0;   /* 模拟元数据：模数链层级（桩不修改它）*/
};

/* ---- 3) CKKS 运算抽象接口（“抽象类”）---- */
template <class T, std::size_t N>
class CkksOps {
public:
    static constexpr std::size_t slots = N;
    virtual ~CkksOps() = default;

    /* 密文域二元/一元同态运算（仅 ct-ct 与 ct-pt 组合）*/
    virtual Ciphertext<T, N> add_cc(const Ciphertext<T, N> &a, const Ciphertext<T, N> &b) = 0;
    virtual Ciphertext<T, N> add_cp(const Ciphertext<T, N> &a, const Plaintext<T, N> &b) = 0;
    virtual Ciphertext<T, N> sub_cc(const Ciphertext<T, N> &a, const Ciphertext<T, N> &b) = 0;
    virtual Ciphertext<T, N> sub_cp(const Ciphertext<T, N> &a, const Plaintext<T, N> &b) = 0;
    virtual Ciphertext<T, N> mul_cc(const Ciphertext<T, N> &a, const Ciphertext<T, N> &b) = 0;
    virtual Ciphertext<T, N> mul_cp(const Ciphertext<T, N> &a, const Plaintext<T, N> &b) = 0;
    virtual Ciphertext<T, N> negate(const Ciphertext<T, N> &a) = 0;
    virtual Ciphertext<T, N> rotate(const Ciphertext<T, N> &a, int k) = 0;   /* 循环旋转 */

    /* CKKS 专有（本期桩：仅记录，不改数据）*/
    virtual void rescale(Ciphertext<T, N> &a) = 0;
    virtual void modswitch(Ciphertext<T, N> &a) = 0;
};

/* ---- 4) 模拟后端：VecOps 实现 CkksOps（用逐元素向量运算模拟同态语义）----
 *
 * 注：返回 Ciphertext<T,N> 按值；模拟后端直接构造具体基类，无切片。
 *     真实 CKKS 可派生 Plaintext/Ciphertext 与 VecOps，在派生引擎里 override
 *     并返回派生类型。
 */
template <class T, std::size_t N>
class VecOps : public CkksOps<T, N> {
public:
    Ciphertext<T, N> add_cc(const Ciphertext<T, N> &a, const Ciphertext<T, N> &b) override {
        return elementwise(a, b, [](T x, T y) { return x + y; });
    }
    Ciphertext<T, N> add_cp(const Ciphertext<T, N> &a, const Plaintext<T, N> &b) override {
        return elementwise(a, b, [](T x, T y) { return x + y; });
    }
    Ciphertext<T, N> sub_cc(const Ciphertext<T, N> &a, const Ciphertext<T, N> &b) override {
        return elementwise(a, b, [](T x, T y) { return x - y; });
    }
    Ciphertext<T, N> sub_cp(const Ciphertext<T, N> &a, const Plaintext<T, N> &b) override {
        return elementwise(a, b, [](T x, T y) { return x - y; });
    }
    Ciphertext<T, N> mul_cc(const Ciphertext<T, N> &a, const Ciphertext<T, N> &b) override {
        return elementwise(a, b, [](T x, T y) { return x * y; });
    }
    Ciphertext<T, N> mul_cp(const Ciphertext<T, N> &a, const Plaintext<T, N> &b) override {
        return elementwise(a, b, [](T x, T y) { return x * y; });
    }

    Ciphertext<T, N> negate(const Ciphertext<T, N> &a) override {
        Ciphertext<T, N> r;
        for (std::size_t i = 0; i < N; ++i) r[i] = -a[i];
        return r;
    }

    /* 循环左移 k（k 归一化到 [0,N)）*/
    Ciphertext<T, N> rotate(const Ciphertext<T, N> &a, int k) override {
        const std::size_t kk = static_cast<std::size_t>(((k % static_cast<int>(N)) + static_cast<int>(N)) % static_cast<int>(N));
        Ciphertext<T, N> r;
        for (std::size_t i = 0; i < N; ++i) r[i] = a[(i + kk) % N];
        return r;
    }

    /* 桩：仅记录，不改数据 */
    void rescale(Ciphertext<T, N> & /*a*/) override {
        std::printf("[VecOps] rescale() stub: no-op\n");
    }
    void modswitch(Ciphertext<T, N> & /*a*/) override {
        std::printf("[VecOps] modswitch() stub: no-op\n");
    }

private:
    /* 逐元素二元运算（a 为密文，结果也是密文；b 可为密文或明文，只要暴露 operator[]）*/
    template <class B, class Op>
    static Ciphertext<T, N> elementwise(const Ciphertext<T, N> &a, const B &b, Op op) {
        Ciphertext<T, N> r;
        for (std::size_t i = 0; i < N; ++i) r[i] = op(a[i], b[i]);
        return r;
    }
};

/* ---- 5) 常用环维别名 ---- */
template <class T> using VecOps32k = VecOps<T, 32768>;
template <class T> using VecOps64k = VecOps<T, 65536>;

#endif /* MPI_TEST_OPERATIONS_HPP */
