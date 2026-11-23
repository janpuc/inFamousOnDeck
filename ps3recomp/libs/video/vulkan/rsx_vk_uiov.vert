#version 450
// Overlay quad batch (rsx_ui_overlay.c, Vulkan path). Positions arrive in virtual layout pixels
// (720 px tall, width follows the frame aspect); push constants carry 2/vw and 2/vh.
layout(location = 0) in vec2 in_pos;
layout(location = 1) in vec2 in_uv;
layout(location = 2) in vec4 in_col;
layout(push_constant) uniform PC { vec2 scale; } pc;
layout(location = 0) out vec2 v_uv;
layout(location = 1) out vec4 v_col;
void main()
{
    gl_Position = vec4(in_pos.x * pc.scale.x - 1.0, in_pos.y * pc.scale.y - 1.0, 0.0, 1.0);
    v_uv = in_uv;
    v_col = in_col;
}
