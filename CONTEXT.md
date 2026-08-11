# CONTEXT — SkillChain 复现项目领域术语表

> 本文件只是术语表。不写实现细节、不当规格书、不当草稿本。
> 论文：Hu et al. 2026, *SkillChain: Closing the Loop on Skill Evolution for Image-Based E-Commerce AI Assistants*（`docs/` 内 PDF）。

## 核心概念

- **Skill**：四元组 s = (d, b, Cs, Od)，声明式的按意图行为规范，非可执行代码。
- **Description (d)**：路由触发边界描述，仅 Stage 2 可修改。
- **Body (b)**：格式/工具/约束规则，仅 Stage 3 可修改。
- **Static Components (Cs)**：Skill 附带的静态知识与示例。
- **Dynamic Operators (Od)**：Skill 声明的推理时可调用工具清单。
- **Skill Bank (Bk)**：带版本的 Skill 集合。

## 意图

- **五类视觉意图**：Exact Match（找同款）/ Multi-Product（多商品分解比较）/ Divergent Rec.（风格发散推荐）/ Encyclopedia（视觉百科）/ Utility Assistance（工具型任务）。
- **边界模糊样本**：话术或图片处于两意图交界的定向合成查询，Stage 2 的核心养料。

## 管线原则

- **单向链**：Stage 2 只改 d、Stage 3 只改 b，修正不回传上游。
- **单调门**：Stage 2 接受条件 F1(Bk+1) ≥ F1(Bk)；Stage 3 接受条件 J̄(Bk+1) ≥ J̄(Bk)；不满足即回滚。
- **Human Reflection Gate**：Skill 入库/修订前的真人终审环节（本项目：LLM 预审报告 + 用户终审）。
- **Engineer Loop**：Stage 1 中对 Skill 草稿的自动校验回炉循环（operators 存在性、静态资源可解析、试跑无结构性失败）。

## 评测

- **四维 Judge**：TCR 工具调用合理性 (0-10) / CCC 卡片编排合规 (0-10，仅对产出商品卡的查询计) / CQ 内容质量 (0-20) / CA 约束遵循 (0-10)；J̄ = 2×四维和，归一化 [0,100]。
- **Routing-Deviation Fallback Principle**：Judge 最高优先原则——路由偏差时机械套模板与拒答同罪，合理偏离规则以服务真实意图应得分。
- **Cross-Sample Attribution**：不对单样本分数行动；按 Skill 聚合 Good/Average/Poor 层级分布，Poor 占比超维度阈值 θd 才触发修复。
- **双路评测**：确定性规则路径（结构检查）+ LLM Judge 路径（四维打分），二者失败模式互补。
- **冻结测试集**：切分后禁止任何写入的最终评测集，四配置可比性与单调门比较的前提。

## 路由优化

- **路由失败三根因**：Boundary ambiguity（边界模糊，主导）/ Missing skill（无 Skill 覆盖）/ Visual parsing error（视觉解析错误，上抛不处理）。
- **Description 三操作**：Update（改写边界）/ Merge（合并相邻 Skill）/ Discard（废弃）。
