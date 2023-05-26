$('#recordButton').addClass("notRec");

$('#recordButton').click(function(){
	if($('#recordButton').hasClass('notRec')){
		$('#recordButton').removeClass("notRec");
		$('#recordButton').addClass("Rec");
	}
	else{
		$('#recordButton').removeClass("Rec");
		$('#recordButton').addClass("notRec");
	}
});	