import torch
from torch import nn


class LMHead(nn.Module):
    """把每个位置的隐藏向量映射回词表上的分数。

    契约（任务 9 把它落成 tests/test_lm_head.py 里的断言）：

      形状   权重 [V, D]；输入 hidden_states [B, S, D] → 输出 logits [B, S, V]
      bias   默认 False，对齐 GPT-2
      构造   hidden_size / vocab_size 不是正数 → ValueError
      输入   不是三维 → ValueError；最后一维 != hidden_size → ValueError
      解码   forward_last_position 只对最后一位做矩阵乘，
             结果与 forward(...)[:, -1, :] 在浮点容差内一致（不是逐比特相等，
             两种形状走不同的 BLAS 归约顺序，GPT-2 尺度实测差到 ~3e-6）
    """

    def __init__(self, hidden_size: int, vocab_size: int, *, bias: bool = False):
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, but got {hidden_size}")
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, but got {vocab_size}")
        self.proj = nn.Linear(hidden_size, vocab_size, bias=bias)

    @property
    def weight(self) -> nn.Parameter:
        """[V, D]，和 TokenEmbedding 的权重形状相同，供 tie_weights() 共享。"""
        return self.proj.weight

    def _validate(self, hidden_states: torch.Tensor) -> None:
        if hidden_states.dim() != 3:
            raise ValueError(
                f"hidden_states must be a 3D tensor [B, S, D], "
                f"but got shape {tuple(hidden_states.shape)}"
            )
        hidden_size = self.proj.in_features
        if hidden_states.size(-1) != hidden_size:
            raise ValueError(
                f"hidden_states last dimension must be {hidden_size}, "
                f"but got shape {tuple(hidden_states.shape)}"
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """全量路径：[B, S, D] → [B, S, V]。"""
        self._validate(hidden_states)
        return self.proj(hidden_states)

    def forward_last_position(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """解码路径：[B, S, D] → [B, V]，只对最后一个位置做矩阵乘。

        切片必须发生在矩阵乘**之前**，否则省不下任何计算。
        """
        self._validate(hidden_states)
        if hidden_states.size(1) == 0:
            raise ValueError(
                f"hidden_states must have at least one position to decode from, "
                f"but got shape {tuple(hidden_states.shape)}"
            )
        # 先切成 [B, D] 再乘，省下前 S-1 个位置的矩阵乘；
        # 反过来写 self.proj(hidden_states)[:, -1, :] 结果一样但一分计算都不省
        return self.proj(hidden_states[:, -1, :])


if __name__ == "__main__":
    # 任务 3 的即时自检。任务 9 会搬进 tests/test_lm_head.py 改成 pytest。
    # 跑法：python src/mini_transformer/lm_head.py
    import time

    D, V = 16, 128

    def check_weight_shape_is_vocab_by_hidden():
        head = LMHead(hidden_size=D, vocab_size=V)
        assert head.weight.shape == (V, D), head.weight.shape

    def check_bias_defaults_to_false():
        assert LMHead(hidden_size=D, vocab_size=V).proj.bias is None
        assert LMHead(hidden_size=D, vocab_size=V, bias=True).proj.bias is not None

    def check_forward_shape():
        head = LMHead(hidden_size=D, vocab_size=V)
        assert head(torch.randn(2, 5, D)).shape == (2, 5, V)

    def check_last_position_shape():
        head = LMHead(hidden_size=D, vocab_size=V)
        assert head.forward_last_position(torch.randn(2, 5, D)).shape == (2, V)

    def check_two_paths_agree():
        # 用 assert_close 而不是 torch.equal：两种输入形状走不同的 BLAS 归约顺序，
        # 浮点加法不满足结合律，GPT-2 尺度上逐比特相等会偶发失败
        for shape in [(2, 5, D), (1, 1, D), (4, 128, D)]:
            head = LMHead(hidden_size=shape[-1], vocab_size=V)
            x = torch.randn(*shape)
            with torch.no_grad():
                torch.testing.assert_close(
                    head.forward_last_position(x), head(x)[:, -1, :]
                )

    def check_single_position_is_consistent():
        # S=1 时两条路径的语义必须重合，这是解码第一步的情形
        head = LMHead(hidden_size=D, vocab_size=V)
        x = torch.randn(3, 1, D)
        with torch.no_grad():
            torch.testing.assert_close(
                head.forward_last_position(x), head(x).squeeze(1)
            )

    def check_rejects_non_positive_hidden_size():
        try:
            LMHead(hidden_size=0, vocab_size=V)
        except ValueError as e:
            assert "hidden_size" in str(e), e
        else:
            raise AssertionError("hidden_size=0 应该抛 ValueError")

    def check_rejects_non_positive_vocab_size():
        try:
            LMHead(hidden_size=D, vocab_size=0)
        except ValueError as e:
            assert "vocab_size" in str(e), e
        else:
            raise AssertionError("vocab_size=0 应该抛 ValueError")

    def check_rejects_non_3d_input():
        head = LMHead(hidden_size=D, vocab_size=V)
        for bad in [torch.randn(5, D), torch.randn(2, 3, 4, D)]:
            for fn in (head, head.forward_last_position):
                try:
                    fn(bad)
                except ValueError as e:
                    # 错误信息必须带上实际形状，否则和 torch 的兜底没区别
                    assert str(tuple(bad.shape)) in str(e), e
                else:
                    raise AssertionError(f"{tuple(bad.shape)} 应该抛 ValueError")

    def check_rejects_wrong_hidden_size():
        head = LMHead(hidden_size=D, vocab_size=V)
        bad = torch.randn(2, 5, D + 1)
        for fn in (head, head.forward_last_position):
            try:
                fn(bad)
            except ValueError as e:
                assert str(D) in str(e) and str(tuple(bad.shape)) in str(e), e
            else:
                raise AssertionError("最后一维不匹配应该抛 ValueError")

    def check_rejects_empty_sequence_when_decoding():
        head = LMHead(hidden_size=D, vocab_size=V)
        try:
            head.forward_last_position(torch.randn(2, 0, D))
        except ValueError as e:
            assert "at least one position" in str(e), e
        else:
            raise AssertionError("S=0 应该抛 ValueError")

    def check_accepts_empty_batch():
        # B=0 是合法的（对齐 embedding.py 允许空 batch），只有 S=0 才拦
        head = LMHead(hidden_size=D, vocab_size=V)
        assert head.forward_last_position(torch.randn(0, 5, D)).shape == (0, V)

    def check_runs_under_inference_mode():
        head = LMHead(hidden_size=D, vocab_size=V)
        with torch.inference_mode():
            out = head.forward_last_position(torch.randn(2, 5, D))
        assert out.requires_grad is False

    torch.manual_seed(0)
    checks = [v for k, v in sorted(locals().items()) if k.startswith("check_")]
    for check in checks:
        check()
        print(f"ok  {check.__name__}")
    print(f"\n{len(checks)} checks passed")

    # 切片在矩阵乘之前才有意义。上面的断言全是数值等价性，抓不到顺序写反，
    # 所以这里用 GPT-2 尺度的耗时把「省下的计算」直接量出来。
    gpt2 = LMHead(hidden_size=768, vocab_size=50257)
    x = torch.randn(1, 1024, 768)

    def timed(fn, n=5):
        with torch.no_grad():
            fn()  # 预热，避免把首次分配算进去
            start = time.perf_counter()
            for _ in range(n):
                fn()
            return (time.perf_counter() - start) / n * 1000

    full_ms = timed(lambda: gpt2(x)[:, -1, :])
    last_ms = timed(lambda: gpt2.forward_last_position(x))
    print(f"\nGPT-2 尺度 [1, 1024, 768] @ [768, 50257]")
    print(f"  先算全量再切片  {full_ms:7.2f} ms   {2 * 1024 * 768 * 50257 / 1e9:6.1f} GFLOP")
    print(f"  先切片再算      {last_ms:7.2f} ms   {2 * 1 * 768 * 50257 / 1e9:6.3f} GFLOP")
    print(f"  加速            {full_ms / last_ms:7.1f}x")
