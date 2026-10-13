#pragma once
#include "runtime/plan.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
namespace fhegpu {
// Storage and temporary peaks, with device foreground and plaintext tracked separately.
struct ResourceUsage {
    std::vector<std::uint64_t> front, plain;
    std::uint64_t host = 0, pinned = 0, raw = 0;
    explicit ResourceUsage(std::size_t devices = 0) : front(devices), plain(devices) {}
    static void add(std::uint64_t &a, std::uint64_t b) {
        if (b > std::numeric_limits<std::uint64_t>::max() - a) throw std::overflow_error("Waterline byte overflow");
        a += b;
    }
    ResourceUsage &operator+=(const ResourceUsage &b) {
        if (front.size() != b.front.size()) throw std::logic_error("Waterline device table mismatch");
        for (std::size_t d=0; d<front.size(); ++d) {add(front[d],b.front[d]);add(plain[d],b.plain[d]);}
        add(host,b.host);add(pinned,b.pinned);add(raw,b.raw);return *this;
    }
    ResourceUsage &operator-=(const ResourceUsage &b) {
        const auto sub=[](std::uint64_t &a,std::uint64_t n) {if(n>a)throw std::logic_error("Waterline credit underflow");a-=n;};
        for (std::size_t d=0; d<front.size(); ++d) {sub(front[d],b.front[d]);sub(plain[d],b.plain[d]);}
        sub(host,b.host);sub(pinned,b.pinned);sub(raw,b.raw);return *this;
    }
};
struct PrepareResources { ResourceUsage peak, resident; bool supported = false; };
class WaterlineLedger {
public:
    explicit WaterlineLedger(const WaterlineConfig &config) : used(config.device_capacity_bytes.size()), peak(used.front.size()), config_(config) {}
    bool fits(const ResourceUsage &delta) const {
        auto next=used;next+=delta;
        for (std::size_t d=0;d<next.front.size();++d) {
            const auto m=config_.device_capacity_bytes[d], k=config_.device_reserve_bytes[d];
            if (next.front[d]>m-k || next.plain[d]>m-k-next.front[d] ||
                next.plain[d]>static_cast<std::uint64_t>(config_.device_fractions.at(d)*m)) return false;
        }
        return next.host<=config_.host_budget_bytes && next.pinned<=config_.pinned_budget_bytes && next.raw<=config_.raw_budget_bytes;
    }
    void reserve(const ResourceUsage &delta) {
        if (!fits(delta)) throw std::logic_error("Waterline reservation without complete group admission");
        used+=delta;
        for (std::size_t d=0;d<used.front.size();++d) {peak.front[d]=std::max(peak.front[d],used.front[d]);peak.plain[d]=std::max(peak.plain[d],used.plain[d]);}
        peak.host=std::max(peak.host,used.host);peak.pinned=std::max(peak.pinned,used.pinned);peak.raw=std::max(peak.raw,used.raw);
    }
    void release(const ResourceUsage &delta) {used-=delta;}
    std::string describe(const ResourceUsage &delta) const {
        auto next=used;next+=delta;std::string result;
        for(std::size_t d=0;d<next.front.size();++d) result+=" GPU"+std::to_string(d)+" front="+std::to_string(next.front[d])+" plain="+std::to_string(next.plain[d])+" M="+std::to_string(config_.device_capacity_bytes[d])+" reserve="+std::to_string(config_.device_reserve_bytes[d]);
        return result+" Host="+std::to_string(next.host)+" pinned="+std::to_string(next.pinned)+" raw="+std::to_string(next.raw);
    }
    ResourceUsage used, peak;
private:
    const WaterlineConfig &config_;
};
} // namespace fhegpu
