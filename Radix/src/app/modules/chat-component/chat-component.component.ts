import { Component } from '@angular/core';

@Component({
  selector: 'app-chat-component',
  templateUrl: './chat-component.component.html',
  styleUrls: ['./chat-component.component.css']
})
export class ChatComponentComponent {
  newMessage: string = '';
  messages: string[] = [];
  date = new Date();
  selectedImage: string | ArrayBuffer | undefined = undefined;
  hour: string = new Date().toLocaleTimeString();

  sendMessage() {
    if (this.newMessage.trim() !== '') {
      this.messages.push(this.newMessage);
      this.newMessage = '';
    }
  }

  readURL(event: Event) {
    const target = event.target as HTMLInputElement;
    if (target && target.files && target.files.length > 0) {
      const file = target.files[0];
      const reader = new FileReader();
      reader.onload = e => {
        if (typeof reader.result === 'string') {
          this.selectedImage = reader.result;
          this.messages.push(this.selectedImage);
        } else if (reader.result) {
          this.selectedImage = reader.result.toString();
          this.messages.push(this.selectedImage);
        }
      };
      reader.readAsDataURL(file);
    }
  }

  isImageUrl(url: string): boolean {
    return url.startsWith('data:image');
  }
}
