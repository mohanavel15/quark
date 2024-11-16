const Self = @This();

var instance: Self = undefined;
var initialized: bool = false;

pub fn GetInstance() Self {
    if (!initialized) {
        @panic("Backend Not Initialized!");
    }

    return instance;
}

context: *anyopaque,

extern fn initialize() *anyopaque;
extern fn deinitialize(context: *anyopaque) void;

pub fn Init() Self {
    const ctx = initialize();
    instance = Self{
        .context = ctx,
    };
    initialized = true;
    return instance;
}

pub fn Deinit(self: *Self) void {
    initialized = false;
    deinitialize(self.context);
}
