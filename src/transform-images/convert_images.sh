#!/bin/bash

# 检查参数数量
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <target_registry_namespace> [dryrun]"
    exit 1
fi

# 目标镜像的 registory 地址加 namespace
target_registry_namespace=$1
echo "target registry namespace is: $target_registry_namespace"

# 是否为 dryrun 模式
dryrun_mode=false
if [ "$#" -eq 2 ] && [ "$2" == "dryrun" ]; then
    dryrun_mode=true
fi

# 读取 image.txt 文件并进行处理
while IFS= read -r line || [[ -n "$line" ]]; do
    # 忽略注释行和空行
    if [[ $line == \#* ]] || [[ -z $line ]]; then
        continue
    fi

    # 提取镜像名和标签
    image_name=$(echo "$line" | cut -d ':' -f 1)
    image_tag=$(echo "$line" | cut -d ':' -f 2)

    # 如果镜像名中不含有 registory 地址，则默认为 Docker Hub 镜像
    if [[ ! $image_name =~ .*/.* ]]; then
        image_name="registry-1.docker.io/$image_name"
    fi

    # 构建目标镜像名
    target_image="$target_registry_namespace/$(basename $image_name):$image_tag"

    # 输出目标镜像名
    echo "target image is: $target_image"

    # 如果不是 dryrun 模式，则执行镜像转换操作
    if ! $dryrun_mode; then
        # 拉取源镜像
        docker pull "$image_name:$image_tag"
        # 重新打标签
        docker tag "$image_name:$image_tag" "$target_image"
        # 推送目标镜像
        docker push "$target_image"
    fi
done < "images.txt"