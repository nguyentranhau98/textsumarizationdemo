var content = document.getElementById("content");
var generated = document.getElementById("generated");
var task = document.getElementById("task");

var send_data = (prompt) => {
    task_ = task.value;

    $.ajax({
        type: 'POST',
        url: '/predict',
        data: JSON.stringify({
            content: prompt,
            task: task_,

        }),
        success: function (msg, status, jqXHR) {
            generated.value = msg;
            console.log(msg);
        },
        contentType: "application/json",
        dataType: 'json'
    });
}
$("#submit").click(function () {
    value = content.value;
    send_data(value);
});