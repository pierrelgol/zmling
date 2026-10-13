//
// A bag that already has a label sometimes needs another one.
// The usual reason: two pieces of code disagree on vocabulary.
// The weights file says `.batch`, your model says `.b`. Or one
// tensor has two axes that mean the same kind of thing and must
// still be told apart (`.from` and `.to`). Later chapters match
// axes BY NAME when combining tensors, so the names had better
// be the ones you intend.
//
// ZML has three ways to relabel. They're not equally careful,
// and Brenda has opinions.
//
//     x.rename(.{ .old = .new, .other_old = .other_new })
//
//         The good one. A struct: field names are the tags to
//         replace, values are the new tags. Axes you don't
//         mention keep theirs. And it CHECKS:
//
//         - the old tag must exist, or:
//             panic: Tensor {a=2,b=3,f32} doesn't have dimension
//                    with tag: zz
//
//         - the new tag must not be in use already, or:
//             panic: {a=2,b=3,f32}.rename(.{ .a = .b }) failed
//                    because of duplicated axis .b
//
//         "In use" means in use in the tensor as it was BEFORE
//         the call. So no, you cannot swap two tags with
//         `.rename(.{ .a = .b, .b = .a })`. It panics too. (To
//         swap, relabel everything with `withTags`.)
//
//     x.renameTag(old, new)
//
//         One axis at a time; `old` may be a tag or an axis
//         number. It does NOT check for duplicates.
//
//     x.renameAxis(axis, .new)
//
//         Same, but the axis is given by number only (negative
//         numbers count from the end, as everywhere). No
//         duplicate check either.
//
// Why does the check matter? Because ZML lets a duplicated tag
// exist: `{bags=3,bags=4,f32}` is a shape it will print for you
// without blinking. Then you ask for `dim(.bags)`, get the FIRST
// one, 3, and no complaint. Two suitcases, one label: every
// lookup after that is a coin toss that always lands on the same
// side. So: `rename` by default. The unchecked ones are for when
// you know something ZML doesn't, typically for putting a first
// label on an axis that has none.
//
// Like the others, these three only touch the Shape. No
// operation, no cost, legal anywhere.
//
// On the desk: the manifest of unclaimed luggage, one row per
// flight, and for each bag its weight and its size. Brenda is
// at lunch. Kevin has been at the label maker: four relabelings,
// four problems. Two panic. One is a blank. One runs, and is the
// worst of the four.
//
// A word about the first panic, because it points at the wrong
// suspect. `rename` looks for duplicates BEFORE it looks for the
// old tags, so a rename written the wrong way around is reported
// as "duplicated axis". Read the whole message, not its last
// word: it prints the shape and the call. Which side of each `=`
// is a tag the manifest really has?
//
const std = @import("std");
const zml = @import("zml");

pub fn main(init: std.process.Init) !void {
    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &buffer);
    const stdout = &stdout_writer.interface;

    // Three flights, four bags per flight, two numbers per bag.
    // The last axis never got a label.
    const manifest = zml.Tensor.init(.{ 3, 4, 2 }, .f32).withTags(.{ .flights, .bags, ._ });
    try stdout.print("Manifest: {f}\n", .{manifest});

    // The customs form calls flights "departures" and bags
    // "items". Both at once, with the careful method.
    const customs = manifest.rename(.{ .flights = .departures, .items = .bags });
    try stdout.print("Customs:  {f}\n", .{customs});

    // Put a first label on the last axis: .measures. It has no
    // tag to rename FROM, so go by position, from the end.
    const complete = customs.renameAxis(???, .measures);
    try stdout.print("Complete: {f}, fully tagged? {}\n", .{ complete, complete.shape().isFullyTagged() });

    // The archive files everything by flight: .departures must be
    // called .flights again. Kevin did this one. Run it, look
    // closely at the shape and at the number of items it reports,
    // then do it the careful way.
    const archive = complete.renameTag(.departures, .items);
    try stdout.print("Archive:  {f}, {d} items per flight.\n", .{ archive, archive.dim(.items) });

    // Transfers: how many bags go from each of 3 flights to each
    // of 5 others. For the trip back, the two labels trade
    // places. Kevin asked the careful method for a swap; the
    // lesson says what it thinks of that, and what to use.
    const transfers = zml.Tensor.init(.{ .from = 3, .to = 5 }, .f32);
    const returns = transfers.rename(.{ .from = .to, .to = .from });
    try stdout.print("Returns:  {f}, back to {d} flights.\n", .{ returns, returns.dim(.to) });
    try stdout.flush();
}
