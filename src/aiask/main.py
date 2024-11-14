import typer
import os
import click
from .config_manager import (
    add_model_to_config,
    remove_model_from_config,
    set_active_model_in_config,
    load_config,
    get_active_model
)
from . import __version__
from rich.table import Table
from rich.console import Console
from rich.markdown import Markdown
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser


app = typer.Typer()
model_app = typer.Typer()
console = Console()


class ShellCommandOutput(BaseModel):
    suggestion: str = Field(description="The suggestion of the command")
    result_command: str = Field(description="The result of the command")


active_model_config = get_active_model()
model = ChatOpenAI(
    model=active_model_config.get('model', ""),
    base_url=active_model_config.get('base_url', ''),
    api_key=active_model_config.get('api_key', ''),
    temperature=active_model_config.get('temperature', 0.2),
)

parser = JsonOutputParser(pydantic_object=ShellCommandOutput)

prompt = PromptTemplate(
    template="你是一个专业的系统管理员。请帮我解决与系统管理相关的问题，在 suggestion 参数中使用 markdown 格式给出具体的命令行和和列举出3~5个的详细参数信息，如果涉及到系统安全的命令，请特别提示，最后综合考虑在 result_command 参数中输出一个最恰当的命令.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

@model_app.command("add")
def add_model(
    name: str = typer.Option(..., prompt=True),
    model: str = typer.Option(..., prompt=True),
    base_url: str = typer.Option(..., prompt=True),
    temperature: float = typer.Option(..., prompt=True),
    api_key: str = typer.Option(..., prompt=True)
):
    """添加新的模型配置"""
    model_config = {
        "model": model,
        "base_url": base_url,
        "temperature": temperature,
        "api_key": api_key
    }
    add_model_to_config(name, model_config)
    typer.echo(f"模型 '{name}' 已添加并激活。")

@model_app.command("remove")
def remove_model(name: str = typer.Option(..., prompt=True)):
    """删除指定的模型配置"""
    remove_model_from_config(name)
    typer.echo(f"模型 '{name}' 已删除。")

@model_app.command("active")
def activate_model(name: str = typer.Option(..., prompt=True)):
    """激活指定的模型配置"""
    set_active_model_in_config(name)
    typer.echo(f"模型 '{name}' 已激活。")

@model_app.command("list")
def list_models():
    """列出所有可用的模型配置"""
    config = load_config()
    active_model = config["active_model"]

    table = Table(title="可用模型配置")
    table.add_column("模型名称", style="cyan")
    table.add_column("状态", style="magenta")

    for name in config["models"]:
        status = "激活" if name == active_model else ""
        table.add_row(name, status)

    console.print(table)

@model_app.command("show")
def show_config():
    """显示当前的模型配置"""
    config = load_config()
    active_model = config["active_model"]

    if not config["models"]:
        typer.echo("当前没有保存的模型配置。请运行 'aigit add-model' 命令添加模型。")
        return

    table = Table(title=f"当前激活的模型配置: {active_model}")
    table.add_column("配置项", style="cyan", no_wrap=True)
    table.add_column("值", style="magenta")

    active_config = config["models"].get(active_model, {})
    for key, value in active_config.items():
        if key == 'api_key' and value:
            value = value[:4] + '*' * (len(value) - 4)
        table.add_row(key, str(value))

    console.print(table)


@app.command()
def msg(question: str):
    """输入问题"""
    if not active_model_config:
        typer.echo("错误：未找到激活的模型配置。请先运行 'aiask model add' 或 'aiask model active' 命令。")
        raise typer.Exit(code=1)

    prompt_msg = f"\n{question}\n",
    chain = prompt | model | parser
    answer = chain.invoke({"query": prompt_msg})

    console.print(Markdown(f"\n**建议:**\n\n{answer['suggestion']}\n\n**命令:**\n\n{answer['result_command']}"))

    if answer["result_command"]:
        execute = typer.confirm(f"\n是否执行该命令: {answer['result_command']}")
        if execute:
            need_edit = typer.confirm("\n是否需要修改命令(默认为 N)?", default=False)
            if need_edit:
                edited_cmd = click.edit(answer["result_command"])
                if edited_cmd is not None:
                    cmd = edited_cmd.strip()
                    if cmd:
                        os.system(cmd)
            else:
                os.system(answer["result_command"])
    else:
        console.print("\n未找到可执行的命令。")

@app.command()
def version():
    """显示当前软件版本"""
    typer.echo(f"AI Ask V{__version__}")


app.add_typer(model_app, name="model", help="管理AI模型")


if __name__ == "__main__":
    app()
