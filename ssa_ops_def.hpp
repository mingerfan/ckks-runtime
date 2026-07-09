#include <cstdint>
#include <tuple>
#include <vector>

// 一般来说，左侧是id，用于标注op序号，右侧是这个op的输出绑定的devices
// 使用vector主要是表明，这个op的数据不一定只在一个设备上，可能同时存在不同的设备上
using id_devices_pair = std::tuple<uint64_t, std::vector<uint32_t>>;

enum SsaOpType {
  ADD_CP,
  ADD_CC,
  MUL_CP,
  MUL_CC,
  ROT,
  RESCALE,
  MODSWITCH,
  BOOT,
};

// Ops是否应该记录操作数的device位置？
// Ops是在什么层级，是干什么用的？
// SsaOps应该是SSA的忠实转译，因此应该表示SSA的元信息，包括OP_TYPE以及所处device位置
// 但是SsaOps不负责真正的内存管理，那应该是runtime应该做的工作
class SsaOps {
  SsaOpType op_type;
  uint64_t id;
  std::vector<id_devices_pair> inputs;
  uint16_t scale;
  uint16_t level;
  uint32_t compute_device;
};
