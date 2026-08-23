# PhishGuard 后端公网部署

本目录是可以直接放进独立 GitHub 仓库并通过 Render Docker Web Service 部署的后端包。

## 安全要求

- 不要把 DeepSeek API Key 写入任何文件或提交到 GitHub。
- `OPENAI_API_KEY` 只在 Render 创建服务时填入 Secret。
- CORS 仅允许 `https://ancien0126.github.io`，不要使用 `*`。
- 建议在 DeepSeek 控制台设置余额预警或消费上限。

## 部署步骤

1. 新建一个 GitHub 仓库，例如 `PHISHGUARD-BACKEND`，把本目录中的内容上传到仓库根目录。
2. 登录 Render，选择 `New` -> `Blueprint`，连接该仓库。
3. Render 会读取根目录中的 `render.yaml`。
4. 创建服务时，在 `OPENAI_API_KEY` 中填入 DeepSeek API Key。
5. 等待构建完成，记录 Render 提供的 HTTPS 地址，例如 `https://phishguard-api.onrender.com`。
6. 访问 `https://你的地址/health`，应返回版本和两个检测器名称。
7. 在前端 `assets/app.js` 中把 `const API_BASE = "";` 改成不带末尾斜杠的后端地址。
8. 前端负责人推送改动，等待 GitHub Pages 更新。

## 联调检查

从前端依次测试文本、网址、HTML 和图片。浏览器开发者工具中不应出现 CORS 错误，检测结果中的 `security-rules-v3` 和 `ai-semantic-v5` 应显示参与状态。

Render 免费服务空闲后会休眠。正式演示或录制前先访问一次 `/health`，等待服务唤醒后再开始检测。
