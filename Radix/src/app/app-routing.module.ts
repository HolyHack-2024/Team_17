import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import {ChatComponentComponent} from "./modules/chat-component/chat-component.component";

const routes: Routes = [
  {path: '', redirectTo: '/chat', pathMatch: 'full'},
  {path: 'chat', component: ChatComponentComponent},
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule {
}
