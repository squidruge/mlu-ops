#!/bin/bash

# 添加文件到暂存区
echo "[status]Adding files to staging area..."
if git add .; then
    echo "[message]Files added successfully."
else
    echo "[message]Failed to add files." >&2
    exit 1
fi

# 提交更改
echo "[status]Committing changes..."
if git commit -m "C2R"; then
    echo "[message]Changes committed successfully."
else
    echo "[message]Failed to commit changes." >&2
    exit 1
fi

# 推送到远程仓库
echo "[status]Pushing changes to remote repository..."
if git push; then
    echo "[message]Changes pushed successfully!!!"
    echo "**********************************************"
else
    echo "[message]Failed to push changes." >&2
    exit 1
fi