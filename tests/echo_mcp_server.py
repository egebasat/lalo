"""A one-tool stdio MCP server for exercising the MCP client. Run: python -m tests.echo_mcp_server"""
from mcp.server.mcpserver import MCPServer

mcp = MCPServer("echo")


@mcp.tool()
def echo(text: str) -> str:
    """Return the text unchanged."""
    return text


if __name__ == "__main__":
    mcp.run()
