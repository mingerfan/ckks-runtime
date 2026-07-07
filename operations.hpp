/* CKKS 同态运算抽象（header-only 模板）。
 *
 * 分三层：
 *   1) Plaintext / Ciphertext 抽象接口
 *      - Plaintext 只暴露明文操作数视图。
 *      - Ciphertext 暴露密文操作数视图，并带 rotate / rescale / modswitch / boot
 *        等密文单算子。
 *   2) VecData<T,N> 具体向量数据，同时实现 Plaintext 与 Ciphertext。
 *   3) CkksOps<T,N>（抽象接口） + VecOps<T,N>（逐元素向量模拟实现）。
 *
 * 设计约束：
 *   - 定长：N 为编译期模板参数（CKKS 环维，2 的幂：32768 / 65536 ...）。
 *   - 只暴露 CKKS 密文域允许的运算：ct⊕ct、ct⊕pt（明文-明文不属于同态运算，不进接口）。
 *   - VecOps 是纯计算封装，不包含通信/传输相关接口。
 *
 *   VecOps<double, 65536> ops;
 *   VecData<double,65536> pt;  pt.fill(2.0);
 *   VecData<double,65536> ct;  ct.fill(3.0);
 *   auto s = ops.add_cp(ct, pt);   // 逐元素 5.0
 *   auto p = ops.mul_cp(ct, pt);   // 逐元素 6.0
 *   auto r = ops.rescale(p);       // 模拟后端为 no-op copy
 */
#ifndef MPI_TEST_OPERATIONS_HPP
#define MPI_TEST_OPERATIONS_HPP

#include <array>
#include <cstddef>

template <class T, std::size_t N>
class VecData;

/* ---- 1) 操作数抽象接口 ---- */
template <class T, std::size_t N>
class Plaintext {
public:
    using value_type = T;
    static constexpr std::size_t slots = N;

    virtual ~Plaintext() = default;

    virtual std::size_t size() const = 0;
    virtual T &operator[](std::size_t i) = 0;
    virtual const T &operator[](std::size_t i) const = 0;
};

template <class T, std::size_t N>
class Ciphertext {
public:
    using value_type = T;
    static constexpr std::size_t slots = N;

    virtual ~Ciphertext() = default;

    virtual std::size_t size() const = 0;
    virtual T &operator[](std::size_t i) = 0;
    virtual const T &operator[](std::size_t i) const = 0;

    virtual int level() const = 0;
    virtual void level(int l) = 0;

    virtual VecData<T, N> negate() const = 0;
    virtual VecData<T, N> rotate(int k) const = 0;
    virtual VecData<T, N> rescale() const = 0;
    virtual VecData<T, N> modswitch() const = 0;
    virtual VecData<T, N> boot() const = 0;
};

/* ---- 2) 定长向量数据：同时可作为明文与密文传入 VecOps ---- */
template <class T, std::size_t N>
class VecData : public Ciphertext<T, N>, public Plaintext<T, N> {
public:
    using value_type = T;
    static constexpr std::size_t slots = N;
    static constexpr std::size_t kSlots = N;

    static_assert(N > 0, "VecData requires at least one slot");

    VecData() = default;

    /* 从迭代器范围构造（长度需 == N，调用方负责）*/
    template <class Iter>
    VecData(Iter first, Iter last) {
        for (std::size_t i = 0; first != last && i < N; ++i, ++first)
            (*this)[i] = static_cast<T>(*first);
    }

    std::size_t size() const override { return N; }

    T &operator[](std::size_t i) override { return data_[i]; }
    const T &operator[](std::size_t i) const override { return data_[i]; }

    void fill(T v) { data_.fill(v); }

    int level() const override { return level_; }
    void level(int l) override { level_ = l; }

    VecData negate() const override {
        VecData r;
        r.level(level_);
        for (std::size_t i = 0; i < N; ++i) r[i] = -data_[i];
        return r;
    }

    /* 循环左移 k（k 归一化到 [0,N)）*/
    VecData rotate(int k) const override {
        const long long n = static_cast<long long>(N);
        long long kk = static_cast<long long>(k) % n;
        if (kk < 0) kk += n;

        VecData r;
        r.level(level_);
        const std::size_t shift = static_cast<std::size_t>(kk);
        for (std::size_t i = 0; i < N; ++i) r[i] = data_[(i + shift) % N];
        return r;
    }

    /* 模拟后端不建模 CKKS 模数链/自举语义，先保持纯计算 no-op copy。 */
    VecData rescale() const override { return copy(); }
    VecData modswitch() const override { return copy(); }
    VecData boot() const override { return copy(); }

private:
    VecData copy() const {
        VecData r;
        r.data_ = data_;
        r.level_ = level_;
        return r;
    }

    std::array<T, N> data_{};
    int level_ = 0;   /* 模拟元数据：模数链层级 */
};

/* ---- 3) CKKS 运算抽象接口（“抽象类”）---- */
template <class T, std::size_t N>
class CkksOps {
public:
    static constexpr std::size_t slots = N;
    virtual ~CkksOps() = default;

    /* 密文域二元/一元同态运算（仅 ct-ct 与 ct-pt 组合）*/
    virtual VecData<T, N> add_cc(const Ciphertext<T, N> &a, const Ciphertext<T, N> &b) = 0;
    virtual VecData<T, N> add_cp(const Ciphertext<T, N> &a, const Plaintext<T, N> &b) = 0;
    virtual VecData<T, N> sub_cc(const Ciphertext<T, N> &a, const Ciphertext<T, N> &b) = 0;
    virtual VecData<T, N> sub_cp(const Ciphertext<T, N> &a, const Plaintext<T, N> &b) = 0;
    virtual VecData<T, N> mul_cc(const Ciphertext<T, N> &a, const Ciphertext<T, N> &b) = 0;
    virtual VecData<T, N> mul_cp(const Ciphertext<T, N> &a, const Plaintext<T, N> &b) = 0;
    virtual VecData<T, N> negate(const Ciphertext<T, N> &a) = 0;
    virtual VecData<T, N> rotate(const Ciphertext<T, N> &a, int k) = 0;
    virtual VecData<T, N> rescale(const Ciphertext<T, N> &a) = 0;
    virtual VecData<T, N> modswitch(const Ciphertext<T, N> &a) = 0;
    virtual VecData<T, N> boot(const Ciphertext<T, N> &a) = 0;
};

/* ---- 4) 模拟后端：VecOps 实现 CkksOps（用逐元素向量运算模拟同态语义）----
 *
 * 注：VecData 同时实现 Plaintext / Ciphertext，因此调用 VecOps 时可直接传入 VecData。
 */
template <class T, std::size_t N>
class VecOps : public CkksOps<T, N> {
public:
    VecData<T, N> add_cc(const Ciphertext<T, N> &a, const Ciphertext<T, N> &b) override {
        return elementwise(a, b, [](T x, T y) { return x + y; });
    }
    VecData<T, N> add_cp(const Ciphertext<T, N> &a, const Plaintext<T, N> &b) override {
        return elementwise(a, b, [](T x, T y) { return x + y; });
    }
    VecData<T, N> sub_cc(const Ciphertext<T, N> &a, const Ciphertext<T, N> &b) override {
        return elementwise(a, b, [](T x, T y) { return x - y; });
    }
    VecData<T, N> sub_cp(const Ciphertext<T, N> &a, const Plaintext<T, N> &b) override {
        return elementwise(a, b, [](T x, T y) { return x - y; });
    }
    VecData<T, N> mul_cc(const Ciphertext<T, N> &a, const Ciphertext<T, N> &b) override {
        return elementwise(a, b, [](T x, T y) { return x * y; });
    }
    VecData<T, N> mul_cp(const Ciphertext<T, N> &a, const Plaintext<T, N> &b) override {
        return elementwise(a, b, [](T x, T y) { return x * y; });
    }

    VecData<T, N> negate(const Ciphertext<T, N> &a) override {
        return a.negate();
    }

    VecData<T, N> rotate(const Ciphertext<T, N> &a, int k) override {
        return a.rotate(k);
    }

    VecData<T, N> rescale(const Ciphertext<T, N> &a) override {
        return a.rescale();
    }

    VecData<T, N> modswitch(const Ciphertext<T, N> &a) override {
        return a.modswitch();
    }

    VecData<T, N> boot(const Ciphertext<T, N> &a) override {
        return a.boot();
    }

private:
    /* 逐元素二元运算（a 为密文，结果也是密文；b 可为密文或明文，只要暴露 operator[]）*/
    template <class B, class Op>
    static VecData<T, N> elementwise(const Ciphertext<T, N> &a, const B &b, Op op) {
        VecData<T, N> r;
        r.level(a.level());
        for (std::size_t i = 0; i < N; ++i) r[i] = op(a[i], b[i]);
        return r;
    }
};

/* ---- 5) 常用环维别名 ---- */
template <class T> using VecOps32k = VecOps<T, 32768>;
template <class T> using VecOps64k = VecOps<T, 65536>;

#endif /* MPI_TEST_OPERATIONS_HPP */
