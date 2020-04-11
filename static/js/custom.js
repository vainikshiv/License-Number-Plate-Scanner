// Ajax call to update function
 $.ajax({
        url: '/update/',
        type: 'POST',
        success: function(json) {
            var a = JSON.parse(json.detect)
            var tr = '<tr><td class="pt-3-half"><img class="card-img-top" style="width: 100px;" src="'+a.image+'" alt="Card image cap"></td><td class="pt-3-half">'+a.number+'</td><td class="pt-3-half" >Car</td><td class="pt-3-half" >Out</td></tr>'
            $(document).find("#update_data").append(tr);
        }
 });