import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './shared/app.component';
import { NavigationComponent } from './shared/navigation/navigation.component';
import { ChatComponentComponent } from './modules/chat-component/chat-component.component';

@NgModule({
  declarations: [
    AppComponent,
    NavigationComponent,
    ChatComponentComponent
  ],
  imports: [
    BrowserModule,
    AppRoutingModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
