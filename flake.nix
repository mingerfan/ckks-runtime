{
  description = "MPI 测试环境（OpenMPI，nixpkgs 来源走国内镜像）";

  # 用国内镜像拉取 nixpkgs 源，避免直接访问 github（国内常被墙/慢）。
  # 首次求值时把当前快照哈希写进 flake.lock，之后换机器/换时间都能复现。
  # 想升级 nixpkgs：nix flake update
  # 备选 input：
  #   https://mirrors.tuna.tsinghua.edu.cn/nix-channels/nixpkgs-unstable/nixexprs.tar.xz
  #   github:NixOS/nixpkgs/nixos-unstable            （国外/有代理时用）
  inputs.nixpkgs.url = "https://mirrors.ustc.edu.cn/nix-channels/nixpkgs-unstable/nixexprs.tar.xz";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
    in {
      devShells.${system}.default = pkgs.mkShell {
        nativeBuildInputs = [ pkgs.openmpi ];

        # 单机测试时允许 -np 超过物理核数
        OMPI_MCA_rmaps_base_oversubscribe = "1";
      };
    };
}
