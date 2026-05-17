原则：
1. 所有改动在legged_gym中进行 
2. 保证不要改动任何leggedgym中原有框架中的代码
3. 新加入的功能都写在一个md文档中，并说明使用/移植方法
4. 为每个新功能建立单独的文件夹，并保证代码简洁高效
5. 测试在conda：rl环境中进行

TODO：
1. 参考pikachu_humanoid_gym中auto_tune和plan_tune的功能，在legged_gym中复现这个功能，UI界面必须复用，和之前的一样，并且将程序保存到legged_gym//legged_gym/tune文件夹中,并在该文件夹中创建使用文档
2. 参考pikachu_humanoid_gym中get_commands_from_keyboard使用键盘控制play的新脚本，并想办法能不能不使用pygame的窗口形式，而是使用更加无感的形式实现键盘控制命令
3. 参考pikachu_gym中projectile的实现，单独创建功能包到utils，要求使用方便简洁，并尝试添加到go2_config中，并在代码中说明用法
 
