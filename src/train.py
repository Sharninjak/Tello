import argparse
from datetime import datetime
from pathlib import Path
import yaml

from ultralytics import YOLO


def get_next_version(base_name: str, check_dir: Path) -> int:
    existing_versions = []
    for d in check_dir.iterdir():
        if d.is_dir() and d.name.startswith(base_name):
            try:
                v_str = d.name.split("_v")[-1]
                existing_versions.append(int(v_str))
            except ValueError:
                pass
    return max(existing_versions, default=0) + 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8 model for cylinder detection")
    parser.add_argument("--model", default="yolov8n.pt", help="Base model path/name")
    parser.add_argument("--data", default="img/cylinder.v1i.yolov8/data.yaml", help="Path to data yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0", help="GPU id or cpu")
    args = parser.parse_args()

    data_yaml = Path(args.data).resolve()
    if not data_yaml.exists():
        raise FileNotFoundError(f"Data yaml not found: {data_yaml}")
    
    # 获取数据集名称和模型名称
    dataset_name = data_yaml.parent.name.split(".")[0]
    model_name = Path(args.model).stem
    
    date_str = datetime.now().strftime("%Y%m%d")
    
    output_root = Path("output").resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    
    # 按照需求构建基础目录名: {数据集}_{模型}_ep{轮数}_{日期}
    base_run_name = f"{dataset_name}_{model_name}_ep{args.epochs}_{date_str}"
    
    # 计算新的自动版本号以避免覆盖
    v_num = get_next_version(base_run_name, output_root)
    run_name = f"{base_run_name}_v{v_num}"
    
    print(f"[*] 解析到数据集: {dataset_name}")
    print(f"[*] 准备输出目录: output/{run_name}")

    model = YOLO(args.model)
    
    # 解析并设置分类名(非常重要, 防止输出为 person 等默认标签)
    with open(data_yaml, 'r', encoding='utf-8') as f:
        data_info = yaml.safe_load(f)
    if 'names' in data_info:
        if isinstance(data_info['names'], list):
            model.model.names = {i: name for i, name in enumerate(data_info['names'])}
        elif isinstance(data_info['names'], dict):
            model.model.names = data_info['names']
        print(f"[*] 修复模型类别映射: {model.model.names}")

    print("\n" + "="*40)
    print(f"[*] 开始训练 {args.epochs} 轮")
    print("="*40)
    
    # 启动训练
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(output_root),
        name=run_name,
        exist_ok=False,
    )

    best_pt = output_root / run_name / "weights" / "best.pt"
    if best_pt.exists():
        print("\n" + "="*40)
        print(f"[*] 训练完成，开始使用最优权重进行测试与验证: {best_pt.name}")
        print("="*40)
        
        # 使用最佳训练权重进行一次评估，存放在同一目录下
        eval_model = YOLO(str(best_pt))
        # 需要再设一次类别，避免 best.pt 未能妥善带过去
        if 'names' in data_info:
            if isinstance(data_info['names'], list):
                eval_model.model.names = {i: name for i, name in enumerate(data_info['names'])}
            elif isinstance(data_info['names'], dict):
                eval_model.model.names = data_info['names']
                
        val_results = eval_model.val(
            data=str(data_yaml),
            split="val",
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=str(output_root),
            name=run_name,
            exist_ok=True, # 允许叠加在原始输出中
        )
        print(f"\n[*] 最终测试验证 mAP50: {val_results.box.map50:.4f}")
        print(f"[*] 最终测试验证 mAP50-95: {val_results.box.map:.4f}")
    else:
        print("[!] 警告: 未找到 best.pt，无法进行后续测试验证。")


if __name__ == '__main__':
    main()
