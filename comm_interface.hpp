#include <cstdint>
#include <optional>
#include <vector>
/*
 * 通信管理抽象层
 * 在执行ssa的过程中，我们面对的算子一般是c=op(a, b)的形式，
 * 然而，要进行运算，我们需要算子a, b在同一个设备上，也就是说，它们的
 * device id应该是同一个。当算子不在同一个设备时，需要启动传输。对于不
 * 同类型的处理单元，比如英伟达的gpu，华为的gpu，加速器之类的，都有不同
 * 的传输api。即使对于同一种设备，因为互联方式的不同，比如8x4卡的集群，
 * 集群内使用nvlink互联，集群间使用网络。这会导致传输处理的混乱，因此需
 * 要一个统一的接口，进行管理。提供抽象的接口。
 */

/*
 * ================= Communication Interface  =================
 */

using device_id = uint32_t;

// 对于host to device传输，我们不将其特殊化
// host会被分配到特定的device，比如device0，
// device0->device1可以理解为host to device，
// device1->device0可以理解为device to host，
// device1->device2则是device间通信
template <typename T> class CommInterface {
  // 传输应该是异步的，当数据传输这个指令发出后，不应该占用时间片来无效的等待
  // 接收函数recive_async主要是用于尽早开始接收
  // recive_fence主要是用于在需要数据时，阻塞的等待需要数据，如果recive_async已经
  // 完成了接收，这个函数就不会阻塞。然而，当数据没有完成传输，这个函数就不会阻塞
  // 等待直到完成传输

  virtual auto send_async(device_id from, device_id to, T &data, uint64_t size,
                          int64_t &transfer_id) -> void = 0;

  virtual auto broadcast_async(device_id from, std::vector<device_id> to, T &data, uint64_t size, int64_t &transfer_id) -> void = 0;
  
  // fence 与 recive_fence共用
  virtual auto gather_async(std::vector<device_id> from, device_id to, T &data, uint64_t size, int64_t &transfer_id) -> void = 0;

  virtual auto recive_async(device_id from, device_id to, int64_t transfer_id)
      -> void = 0;

  virtual auto recive_fence(device_id from, device_id to, int64_t transfer_id)
      -> std::optional<T> = 0;
};
