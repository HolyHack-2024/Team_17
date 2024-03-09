import { Injectable } from '@angular/core';
import {BehaviorSubject, Observable} from "rxjs";

@Injectable({
  providedIn: 'root'
})
export class NavigationService {
  private showNavbar = new BehaviorSubject<boolean>(true);
  public showNavbar$ = this.showNavbar.asObservable();

  public setShowNavbar(show: boolean) {
    this.showNavbar.next(show);
  }

  constructor() { }
}
